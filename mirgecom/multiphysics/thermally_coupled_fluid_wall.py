r"""Operator for thermally-coupled fluid and wall.

Couples a fluid subdomain governed by the compressible Navier-Stokes equations
(:module:`mirgecom.navierstokes) with a wall subdomain governed by the heat
equation (:module:`mirgecom.diffusion`) by enforcing continuity of temperature
and heat flux across their interface.

.. autofunction:: coupled_grad_t_operator
.. autofunction:: get_interface_boundaries
.. autofunction:: coupled_ns_heat_operator

.. autoclass:: InterfaceFluidBoundary
.. autoclass:: InterfaceWallBoundary
"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from dataclasses import replace
import numpy as np

from pytools.obj_array import make_obj_array

from arraycontext import thaw

from grudge.trace_pair import (
    TracePair,
    inter_volume_trace_pairs
)
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    as_dofdesc,
)

from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.fluid import make_conserved
from mirgecom.flux import gradient_flux_central
from mirgecom.inviscid import inviscid_flux_rusanov
from mirgecom.viscous import viscous_flux_central
from mirgecom.gas_model import (
    make_fluid_state,
    make_operator_fluid_states,
)
from mirgecom.navierstokes import (
    grad_t_operator as fluid_grad_t_operator,
    ns_operator,
)
from mirgecom.artificial_viscosity import av_laplacian_operator
from mirgecom.diffusion import (
    DiffusionBoundary,
    grad_operator as wall_grad_t_operator,
    diffusion_operator,
)


class _InterVolNoGradTag:
    pass


class _InterVolTag:
    pass


class InterfaceFluidBoundary(PrescribedFluidBoundary):
    """Interface boundary condition for fluid side."""

    # FIXME: Incomplete docs
    def __init__(self, ext_t, ext_grad_t, ext_kappa):
        """Initialize InterfaceFluidBoundary."""
        PrescribedFluidBoundary.__init__(
            self,
            boundary_state_func=self.get_external_state,
            boundary_grad_av_func=self.get_external_grad_av,
            boundary_temperature_func=self.get_external_t,
            boundary_gradient_temperature_func=self.get_external_grad_t
        )
        self.ext_t = ext_t
        self.ext_grad_t = ext_grad_t
        self.ext_kappa = ext_kappa

    # FIXME: This probably uses the wrong BC for the species mass fractions
    def get_external_state(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior solution on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            ext_t = discr.project(dd_bdry_base, dd_bdry, self.ext_t)
            ext_kappa = discr.project(dd_bdry_base, dd_bdry, self.ext_kappa)
        else:
            ext_t = self.ext_t
            ext_kappa = self.ext_kappa

        # Grab some boundary-relevant data
        dim = discr.dim

        # Cancel out the momentum
        cv_minus = state_minus.cv
        ext_mom = -cv_minus.momentum

        # Compute the energy
        ext_internal_energy = (
            cv_minus.mass
            * gas_model.eos.get_internal_energy(
                temperature=ext_t,
                species_mass_fractions=cv_minus.species_mass_fractions))
        ext_kinetic_energy = gas_model.eos.kinetic_energy(cv_minus)
        ext_energy = ext_internal_energy + ext_kinetic_energy

        # Form the external boundary solution with the new momentum and energy
        ext_cv = make_conserved(
            dim=dim, mass=cv_minus.mass, energy=ext_energy, momentum=ext_mom,
            species_mass=cv_minus.species_mass)
        t_seed = state_minus.temperature if state_minus.is_mixture else None

        def replace_thermal_conductivity(state, kappa):
            new_tv = replace(state.tv, thermal_conductivity=kappa)
            return replace(state, tv=new_tv)

        return replace_thermal_conductivity(
            make_fluid_state(
                cv=ext_cv, gas_model=gas_model, temperature_seed=t_seed),
            ext_kappa)

    # FIXME: This probably uses the wrong BC for the species mass fractions
    def get_external_grad_av(self, discr, dd_bdry, grad_av_minus, **kwargs):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        actx = grad_av_minus.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(discr.normal(dd_bdry), actx)

        # Apply a Neumann condition on the energy gradient
        ext_grad_energy = (
            grad_av_minus.energy - 2 * np.dot(grad_av_minus.energy, nhat) * nhat)

        return make_conserved(
            discr.dim, mass=grad_av_minus.mass, energy=ext_grad_energy,
            momentum=grad_av_minus.momentum, species_mass=grad_av_minus.species_mass)

    def get_external_t(self, discr, dd_bdry, gas_model, state_minus, **kwargs):
        """Get the exterior T on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return discr.project(dd_bdry_base, dd_bdry, self.ext_t)
        else:
            return self.ext_t

    def get_external_grad_t(
            self, discr, dd_bdry, gas_model, state_minus, grad_cv_minus,
            grad_t_minus, **kwargs):
        """Get the exterior grad(T) on the boundary."""
        if dd_bdry.discretization_tag is not DISCR_TAG_BASE:
            dd_bdry_base = dd_bdry.with_discr_tag(DISCR_TAG_BASE)
            return discr.project(dd_bdry_base, dd_bdry, self.ext_grad_t)
        else:
            return self.ext_grad_t


class InterfaceWallBoundary(DiffusionBoundary):
    """Interface boundary condition for wall side."""

    # FIXME: Incomplete docs
    def __init__(self, u_ext, grad_u_ext, kappa_ext):
        """Initialize InterfaceWallBoundary."""
        self.u_ext = u_ext
        self.grad_u_ext = grad_u_ext
        self.kappa_ext = kappa_ext

    def get_grad_flux(
            self, discr, dd_vol, dd_bdry, u, *,
            quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get the numerical flux for grad(u) on the boundary."""
        u_int = discr.project(dd_vol, dd_bdry, u)
        u_tpair = TracePair(dd_bdry, interior=u_int, exterior=self.u_ext)
        from mirgecom.diffusion import grad_flux
        return grad_flux(discr, u_tpair, quadrature_tag=quadrature_tag)

    def get_diffusion_flux(
            self, discr, dd_vol, dd_bdry, u, kappa, grad_u, *,
            penalty_amount=None, quadrature_tag=DISCR_TAG_BASE):  # noqa: D102
        """Get the numerical flux for the diff(u) on the boundary."""
        u_int = discr.project(dd_vol, dd_bdry, u)
        u_tpair = TracePair(dd_bdry, interior=u_int, exterior=self.u_ext)
        kappa_int = discr.project(dd_vol, dd_bdry, kappa)
        kappa_tpair = TracePair(dd_bdry, interior=kappa_int, exterior=self.kappa_ext)
        grad_u_int = discr.project(dd_vol, dd_bdry, grad_u)
        grad_u_tpair = TracePair(
            dd_bdry, interior=grad_u_int, exterior=self.grad_u_ext)
        # Memoized, so should be OK to call here
        from grudge.dt_utils import characteristic_lengthscales
        lengthscales = (
            characteristic_lengthscales(u.array_context, discr, dd_vol) * (0*u+1))
        lengthscales_int = discr.project(dd_vol, dd_bdry, lengthscales)
        lengthscales_tpair = TracePair(
            dd_bdry, interior=lengthscales_int, exterior=lengthscales_int)
        from mirgecom.diffusion import diffusion_flux
        return diffusion_flux(
            discr, u_tpair, kappa_tpair, grad_u_tpair, lengthscales_tpair,
            penalty_amount=penalty_amount, quadrature_tag=quadrature_tag)


def _get_interface_boundaries_no_grad(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature):
    pairwise_vol_data = {
        (fluid_volume_dd, wall_volume_dd): (
            make_obj_array([
                fluid_state.temperature,
                fluid_state.thermal_conductivity]),
            make_obj_array([
                wall_temperature,
                wall_model.thermal_conductivity]))}
    inter_vol_tpairs = inter_volume_trace_pairs(
        discr, pairwise_vol_data, comm_tag=_InterVolNoGradTag)

    fluid_interface_boundaries_no_grad = {}
    fluid_tpairs_no_grad = inter_vol_tpairs[wall_volume_dd, fluid_volume_dd]
    for tpair in fluid_tpairs_no_grad:
        bdtag = tpair.dd.domain_tag
        ext_temperature, ext_kappa = tpair.ext
        fluid_interface_boundaries_no_grad[bdtag] = InterfaceFluidBoundary(
            ext_temperature,
            (0*ext_temperature,)*discr.dim,
            ext_kappa)

    wall_interface_boundaries_no_grad = {}
    wall_tpairs_no_grad = inter_vol_tpairs[fluid_volume_dd, wall_volume_dd]
    for tpair in wall_tpairs_no_grad:
        bdtag = tpair.dd.domain_tag
        ext_temperature, ext_kappa = tpair.ext
        wall_interface_boundaries_no_grad[bdtag] = InterfaceWallBoundary(
            ext_temperature,
            (0*ext_temperature,)*discr.dim,
            ext_kappa)

    return fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad


def coupled_grad_t_operator(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature, *,
        time=0.,
        fluid_numerical_flux_func=gradient_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _fluid_operator_states_quad=None,
        _fluid_interface_boundaries_no_grad=None,
        _wall_interface_boundaries_no_grad=None):
    # FIXME: Incomplete docs
    """Compute grad(T) of the coupled fluid-wall system."""
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    if (
            _fluid_interface_boundaries_no_grad is None
            or _wall_interface_boundaries_no_grad is None):
        fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
            _get_interface_boundaries_no_grad(
                discr,
                gas_model, wall_model,
                fluid_volume_dd, wall_volume_dd,
                fluid_boundaries, wall_boundaries,
                fluid_state, wall_temperature)
    else:
        fluid_interface_boundaries_no_grad = _fluid_interface_boundaries_no_grad
        wall_interface_boundaries_no_grad = _wall_interface_boundaries_no_grad

    fluid_full_boundaries_no_grad = {}
    fluid_full_boundaries_no_grad.update(fluid_boundaries)
    fluid_full_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    wall_full_boundaries_no_grad = {}
    wall_full_boundaries_no_grad.update(wall_boundaries)
    wall_full_boundaries_no_grad.update(wall_interface_boundaries_no_grad)

    return (
        fluid_grad_t_operator(
            discr, gas_model, fluid_full_boundaries_no_grad, fluid_state,
            time=time, numerical_flux_func=fluid_numerical_flux_func,
            quadrature_tag=quadrature_tag, volume_dd=fluid_volume_dd,
            operator_states_quad=_fluid_operator_states_quad),
        wall_grad_t_operator(
            discr, wall_full_boundaries_no_grad, wall_temperature,
            quadrature_tag=quadrature_tag, volume_dd=wall_volume_dd))


def get_interface_boundaries(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature, *,
        time=0.,
        fluid_gradient_numerical_flux_func=gradient_flux_central,
        quadrature_tag=DISCR_TAG_BASE,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _fluid_operator_states_quad=None,
        _fluid_interface_boundaries_no_grad=None,
        _wall_interface_boundaries_no_grad=None):
    # FIXME: Incomplete docs
    """Get the BCs for the interface surfaces between the fluid and the wall."""
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    fluid_grad_temperature, wall_grad_temperature = coupled_grad_t_operator(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature,
        time=time,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        quadrature_tag=DISCR_TAG_BASE,
        _fluid_operator_states_quad=_fluid_operator_states_quad,
        _fluid_interface_boundaries_no_grad=_fluid_interface_boundaries_no_grad,
        _wall_interface_boundaries_no_grad=_wall_interface_boundaries_no_grad)

    pairwise_vol_data = {
        (fluid_volume_dd, wall_volume_dd): (
            make_obj_array([
                fluid_state.temperature,
                fluid_grad_temperature,
                fluid_state.thermal_conductivity]),
            make_obj_array([
                wall_temperature,
                wall_grad_temperature,
                wall_model.thermal_conductivity]))}
    inter_vol_tpairs = inter_volume_trace_pairs(
        discr, pairwise_vol_data, comm_tag=_InterVolTag)

    fluid_interface_boundaries = {}
    fluid_tpairs = inter_vol_tpairs[wall_volume_dd, fluid_volume_dd]
    for tpair in fluid_tpairs:
        bdtag = tpair.dd.domain_tag
        ext_temperature, ext_grad_temperature, ext_kappa = tpair.ext
        fluid_interface_boundaries[bdtag] = InterfaceFluidBoundary(
            ext_temperature,
            ext_grad_temperature,
            ext_kappa)

    wall_interface_boundaries = {}
    wall_tpairs = inter_vol_tpairs[fluid_volume_dd, wall_volume_dd]
    for tpair in wall_tpairs:
        bdtag = tpair.dd.domain_tag
        ext_temperature, ext_grad_temperature, ext_kappa = tpair.ext
        wall_interface_boundaries[bdtag] = InterfaceWallBoundary(
            ext_temperature,
            ext_grad_temperature,
            ext_kappa)

    return fluid_interface_boundaries, wall_interface_boundaries


def _heat_operator(
        discr, wall_model, boundaries, temperature, *,
        penalty_amount, quadrature_tag, volume_dd):
    return (
        1/(wall_model.density * wall_model.heat_capacity)
        * diffusion_operator(
            discr, wall_model.thermal_conductivity, boundaries, temperature,
            penalty_amount=penalty_amount, quadrature_tag=quadrature_tag,
            volume_dd=volume_dd))


def coupled_ns_heat_operator(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature, *,
        time=0., wall_time_scale=1,
        fluid_gradient_numerical_flux_func=gradient_flux_central,
        inviscid_numerical_flux_func=inviscid_flux_rusanov,
        viscous_numerical_flux_func=viscous_flux_central,
        use_av=False,
        av_kwargs=None,
        wall_penalty_amount=None,
        quadrature_tag=DISCR_TAG_BASE):
    # FIXME: Incomplete docs
    """Compute RHS of the coupled fluid-wall system."""
    fluid_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in fluid_boundaries.items()}
    wall_boundaries = {
        as_dofdesc(bdtag).domain_tag: bdry
        for bdtag, bdry in wall_boundaries.items()}

    fluid_interface_boundaries_no_grad, wall_interface_boundaries_no_grad = \
        _get_interface_boundaries_no_grad(
            discr,
            gas_model, wall_model,
            fluid_volume_dd, wall_volume_dd,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_temperature)

    fluid_full_boundaries_no_grad = {}
    fluid_full_boundaries_no_grad.update(fluid_boundaries)
    fluid_full_boundaries_no_grad.update(fluid_interface_boundaries_no_grad)

    fluid_operator_states_quad = make_operator_fluid_states(
        discr, fluid_state, gas_model, fluid_full_boundaries_no_grad,
        quadrature_tag, volume_dd=fluid_volume_dd)

    fluid_interface_boundaries, wall_interface_boundaries = \
        get_interface_boundaries(
            discr,
            gas_model, wall_model,
            fluid_volume_dd, wall_volume_dd,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_temperature,
            time=time,
            fluid_gradient_numerical_flux_func=fluid_gradient_numerical_flux_func,
            quadrature_tag=quadrature_tag,
            _fluid_operator_states_quad=fluid_operator_states_quad,
            _fluid_interface_boundaries_no_grad=fluid_interface_boundaries_no_grad,
            _wall_interface_boundaries_no_grad=wall_interface_boundaries_no_grad)

    fluid_full_boundaries = {}
    fluid_full_boundaries.update(fluid_boundaries)
    fluid_full_boundaries.update(fluid_interface_boundaries)

    wall_full_boundaries = {}
    wall_full_boundaries.update(wall_boundaries)
    wall_full_boundaries.update(wall_interface_boundaries)





    fluid_grad_t, wall_grad_t = coupled_grad_t_operator(
        discr,
        gas_model, wall_model,
        fluid_volume_dd, wall_volume_dd,
        fluid_boundaries, wall_boundaries,
        fluid_state, wall_temperature,
        time=time,
        fluid_numerical_flux_func=fluid_gradient_numerical_flux_func,
        quadrature_tag=quadrature_tag,
        # Added to avoid repeated computation
        # FIXME: See if there's a better way to do this
        _fluid_operator_states_quad=fluid_operator_states_quad,
        _fluid_interface_boundaries_no_grad=fluid_interface_boundaries,
        _wall_interface_boundaries_no_grad=wall_interface_boundaries)





#     fluid_rhs = 0*fluid_grad_t[0]*fluid_grad_t[1]*fluid_state.cv
    fluid_rhs = 0*fluid_grad_t[0]*fluid_grad_t[1] + ns_operator(
        discr, gas_model, fluid_state, fluid_full_boundaries,
        time=time, quadrature_tag=quadrature_tag, volume_dd=fluid_volume_dd,
        operator_states_quad=fluid_operator_states_quad)
#     fluid_rhs = ns_operator(
#         discr, gas_model, fluid_state, fluid_full_boundaries,
#         time=time, quadrature_tag=quadrature_tag, volume_dd=fluid_volume_dd,
#         operator_states_quad=fluid_operator_states_quad)

#     if use_av:
#         if av_kwargs is None:
#             av_kwargs = {}
#         fluid_rhs = fluid_rhs + (0*fluid_grad_t[0]*fluid_grad_t[1] + av_laplacian_operator(
#             discr, fluid_full_boundaries, fluid_state, quadrature_tag=quadrature_tag,
#             volume_dd=fluid_volume_dd, **av_kwargs))

#     wall_rhs = 0*wall_grad_t[0]*wall_grad_t[1]*wall_temperature
    wall_rhs = 0*wall_grad_t[0]*wall_grad_t[1] + wall_time_scale * _heat_operator(
        discr, wall_model, wall_full_boundaries, wall_temperature,
        penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
        volume_dd=wall_volume_dd)
#     wall_rhs = wall_time_scale * _heat_operator(
#         discr, wall_model, wall_full_boundaries, wall_temperature,
#         penalty_amount=wall_penalty_amount, quadrature_tag=quadrature_tag,
#         volume_dd=wall_volume_dd)

    return fluid_rhs, wall_rhs
