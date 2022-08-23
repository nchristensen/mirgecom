r"""
:mod:`mirgecom.transport` provides methods/utils for transport properties.

Transport Models
^^^^^^^^^^^^^^^^
This module is designed provide Transport Model objects used to compute and
manage the transport properties in viscous flows.  The transport properties
currently implemented are the dynamic viscosity ($\mu$), the bulk viscosity
($\mu_{B}$), the thermal conductivity ($\kappa$), and the species diffusivities
($d_{\alpha}$).

.. autoclass:: GasTransportVars
.. autoclass:: TransportModel
.. autoclass:: SimpleTransport
.. autoclass:: PowerLawTransport
.. autoclass:: MixtureAveragedTransport

Exceptions
^^^^^^^^^^
.. autoexception:: TransportModelError
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

from typing import Optional
from dataclasses import dataclass
from arraycontext import dataclass_array_container
import numpy as np
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from mirgecom.fluid import ConservedVars
from mirgecom.eos import GasEOS, GasDependentVars


class TransportModelError(Exception):
    """Indicate that transport model is required for model evaluation."""

    pass


@dataclass_array_container
@dataclass(frozen=True)
class GasTransportVars:
    """State-dependent quantities for :class:`TransportModel`.

    Prefer individual methods for model use, use this
    structure for visualization or probing.

    .. attribute:: bulk_viscosity
    .. attribute:: viscosity
    .. attribute:: volume_viscosity
    .. attribute:: thermal_conductivity
    .. attribute:: species_diffusivity
    """

    bulk_viscosity: np.ndarray
    viscosity: np.ndarray
    volume_viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    species_diffusivity: np.ndarray


class TransportModel:
    r"""Abstract interface to thermo-diffusive transport model class.

    Transport model classes are responsible for
    computing relations between fluid or gas state variables and
    thermo-diffusive transport properties for those fluids.

    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: thermal_conductivity
    .. automethod:: species_diffusivity
    .. automethod:: volume_viscosity
    .. automethod:: transport_vars
    """

    def bulk_viscosity(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas (${\mu}_{B}$)."""
        raise NotImplementedError()

    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        raise NotImplementedError()

    def volume_viscosity(self, cv: ConservedVars,
                         dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the 2nd coefficent of viscosity, $\lambda$."""
        raise NotImplementedError()

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        raise NotImplementedError()

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        raise NotImplementedError()

    def transport_vars(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None,
                       eos: Optional[GasEOS] = None) -> GasTransportVars:
        r"""Compute the transport properties from the conserved state."""
        return GasTransportVars(
            bulk_viscosity=self.bulk_viscosity(cv=cv, dv=dv),
            viscosity=self.viscosity(cv=cv, dv=dv),
            volume_viscosity=self.volume_viscosity(cv=cv, dv=dv),
            thermal_conductivity=self.thermal_conductivity(cv=cv, dv=dv, eos=eos),
            species_diffusivity=self.species_diffusivity(cv=cv, dv=dv, eos=eos)
        )


class SimpleTransport(TransportModel):
    r"""Transport model with uniform, constant properties.

    Inherits from (and implements) :class:`TransportModel`.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self, bulk_viscosity=0, viscosity=0,
                 thermal_conductivity=0,
                 species_diffusivity=None):
        """Initialize uniform, constant transport properties."""
        if species_diffusivity is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._mu_bulk = bulk_viscosity
        self._mu = viscosity
        self._kappa = thermal_conductivity
        self._d_alpha = species_diffusivity

    def bulk_viscosity(self, cv: ConservedVars,
                       dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$."""
        return self._mu_bulk*(0*cv.mass + 1.0)

    def viscosity(self, cv: ConservedVars,
                  dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        return self._mu*(0*cv.mass + 1.0)

    def volume_viscosity(self, cv: ConservedVars,
                         dv: Optional[GasDependentVars] = None) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\mu_{B} - \frac{2\mu}{3}\right)

        """
        return (self._mu_bulk - 2 * self._mu / 3)*(0*cv.mass + 1.0)

    def thermal_conductivity(self, cv: ConservedVars,
                             dv: Optional[GasDependentVars] = None,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$."""
        return self._kappa*(0*cv.mass + 1.0)

    def species_diffusivity(self, cv: ConservedVars,
                            dv: Optional[GasDependentVars] = None,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$."""
        return self._d_alpha*(0*cv.mass + 1.0)


class PowerLawTransport(TransportModel):
    r"""Transport model with simple power law properties.

    Inherits from (and implements) :class:`TransportModel` based on a
    temperature-dependent power law.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    # air-like defaults here
    def __init__(self, scaling_factor=1.0, alpha=0.6, beta=4.093e-7, sigma=2.5,
                 n=.666, species_diffusivity=None, lewis=None):
        """Initialize power law coefficients and parameters.

        factor: Scaling factor to artifically increase or decrease the
        transport coefficients. The default is to keep the physical value, i.e., 1.0.

        lewis: if required, the Lewis number specify the relation between the
        thermal conductivity and the species diffusivities.
        """
        if species_diffusivity is None and lewis is None:
            species_diffusivity = np.empty((0,), dtype=object)
        self._scaling_factor = scaling_factor
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._n = n
        self._d_alpha = species_diffusivity
        self._lewis = lewis

    def bulk_viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$.

        .. math::

            \mu_{B} = \alpha\mu

        """
        return self._alpha * self.viscosity(cv, dv)

    # TODO: Should this be memoized? Avoid multiple calls?
    def viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$.

        $\mu = \beta{T}^n$
        """
        return self._scaling_factor * self._beta * dv.temperature**self._n

    def volume_viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\alpha - \frac{2}{3}\right)\mu

        """
        return (self._alpha - 2.0/3.0)*self.viscosity(cv, dv)

    def thermal_conductivity(self, cv: ConservedVars, dv: GasDependentVars,
                             eos: GasEOS) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$.

        .. math::

            \kappa = \sigma\mu{C}_{v}

        """
        return (
            self._sigma * self.viscosity(cv, dv)
            * eos.heat_capacity_cv(cv, dv.temperature)
        )

    def species_diffusivity(self, cv: ConservedVars, dv: GasDependentVars,
                            eos: GasEOS) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$.

        The species diffusivities can be specified directly or based on the
        user-imposed Lewis number $Le$ of the mixture and the heat capacity at
        constant pressure $C_p$:

        .. math::

            d_{\alpha} = \frac{\kappa}{\rho \; Le \; C_p}
        """
        if self._lewis is not None:
            return (self.thermal_conductivity(cv, dv, eos)/(
                cv.mass*self._lewis*eos.heat_capacity_cp(cv, dv.temperature))
            )
        return self._d_alpha*(0*cv.mass + 1.)


class MixtureAveragedTransport(TransportModel):
    r"""Transport model with mixture averaged transport properties.

    Inherits from (and implements) :class:`TransportModel` based on a
    temperature-dependent fit from Pyrometheus/Cantera weighted by the mixture
    composition.

    .. automethod:: __init__
    .. automethod:: bulk_viscosity
    .. automethod:: viscosity
    .. automethod:: volume_viscosity
    .. automethod:: species_diffusivity
    .. automethod:: thermal_conductivity
    """

    def __init__(self, pyrometheus_mech, alpha=0.6,
                     factor=1.0, prandtl=None, lewis=None):
        r"""Initialize power law coefficients and parameters.

        pyrometheus_mech: The mechanism containg the species properties.

        alpha: The bulk viscosity. The default value is "air". Ideally, it should be
        a function of temperature and species.

        factor: Scaling factor to artifically increase or decrease the
        transport coefficients. The default is to keep the physical value, i.e., 1.0.

        prandtl: if required, the Prandtl number specify the relation between the
        fluid viscosity and the thermal conductivity.

        lewis: if required, the Lewis number specify the relation between the
        thermal conductivity and the species diffusivities. This should be an array.
        """
        self._pyro_mech = pyrometheus_mech
        self._alpha = alpha
        self._factor = factor
        self._prandtl = prandtl
        self._lewis = lewis
        if self._lewis is not None:
            if (len(self._lewis) != self._pyro_mech.num_species):
                raise ValueError("Lewis number should match number of species")

    def viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the gas dynamic viscosity, $\mu$."""
        return (
            self._factor*self._pyro_mech.get_mixture_viscosity_mixavg(
                dv.temperature, cv.species_mass_fractions)
        )

    def bulk_viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the bulk viscosity for the gas, $\mu_{B}$.

        .. math::

            \mu_{B} = \alpha\mu

        """
        return self._alpha*self.viscosity(cv, dv)

    def volume_viscosity(self, cv: ConservedVars, dv: GasDependentVars) -> DOFArray:
        r"""Get the 2nd viscosity coefficent, $\lambda$.

        In this transport model, the second coefficient of viscosity is defined as:

        .. math::

            \lambda = \left(\alpha - \frac{2}{3}\right)\mu

        """
        return (self._alpha - 2.0/3.0)*self.viscosity(cv, dv)

    def thermal_conductivity(self, cv: ConservedVars, dv: GasDependentVars,
                             eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the gas thermal_conductivity, $\kappa$.

        The thermal conductivity can be obtained directly from Pyrometheus using a
        mixture averaged rule considering all species or based on the user-imposed
        Prandtl number of the mixture $Pr$ and the heat capacity at constant pressure
        $C_p$:

        .. math::

            \kappa = \frac{\mu C_p}{Pr}

        """
        if self._prandtl is not None:
            return 1.0/self._prandtl*(
                eos.heat_capacity_cp(cv, dv.temperature)*self.viscosity(cv, dv))
        return self._factor*(self._pyro_mech.get_mixture_thermal_conductivity_mixavg(
            dv.temperature, cv.species_mass_fractions,))

    def species_diffusivity(self, cv: ConservedVars, dv: GasDependentVars,
                            eos: Optional[GasEOS] = None) -> DOFArray:
        r"""Get the vector of species diffusivities, ${d}_{\alpha}$.

        The species diffusivities can be obtained directly from Pyrometheus using a
        mixture averaged rule considering all species or based on the user-imposed
        Lewis number $Le$ of the mixture and the heat capacity at constant pressure
        $C_p$:

        .. math::

            d_{\alpha} = \frac{\kappa}{\rho \; Le \; C_p}

        """
        if self._lewis is not None:
            return (self.thermal_conductivity(cv, dv, eos)/(
                cv.mass*self._lewis*eos.heat_capacity_cp(cv, dv.temperature))
            )
        return self._factor*(self._pyro_mech.get_species_mass_diffusivities_mixavg(
            dv.temperature, dv.pressure, cv.species_mass_fractions))
