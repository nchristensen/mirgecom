"""Test wall-model related functions."""

__copyright__ = """Copyright (C) 2023 University of Illinois Board of Trustees"""

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

import numpy as np
from pytools.obj_array import make_obj_array
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)


def test_tacot_decomposition():
    """Check the wall degradation model."""
    temperature = 900.0

    from mirgecom.materials.tacot import Pyrolysis
    decomposition = Pyrolysis()
    chi = np.array([30.0, 90.0, 160.0])

    tol = 1e-8

    tacot_decomp = decomposition.get_decomposition_parameters()

    print(tacot_decomp)

    # virgin_mass = tacot_decomp["virgin_mass"]
    # char_mass = tacot_decomp["char_mass"]
    # fiber_mass = tacot_decomp["fiber_mass"]
    weights = tacot_decomp["reaction_weights"]
    pre_exp = tacot_decomp["pre_exponential"]
    Tcrit = tacot_decomp["temperature"]  # noqa N806

    # The density parameters are hard-coded for TACOT, depending on
    # virgin and char volume fractions.
    w1 = weights[0]*(chi[0]/(weights[0]))**3
    w2 = weights[1]*(chi[1]/(weights[1]) - 2./3.)**3

    solid_mass_rhs = make_obj_array([
        # reaction 1
        np.where(np.less(temperature, Tcrit[0]),
            0.0, (-w1 * pre_exp[0] * np.exp(-8556.000/temperature))),
        # reaction 2
        np.where(np.less(temperature, Tcrit[1]),
            0.0, (-w2 * pre_exp[1] * np.exp(-20444.44/temperature))),
        # fiber oxidation: include in the RHS but don't do anything with it.
        np.zeros_like(temperature)])

    sample_source_gas = -sum(solid_mass_rhs)

    assert solid_mass_rhs[0] + 26.7676118539965 < tol
    assert solid_mass_rhs[1] + 2.03565420370596 < tol
    assert sample_source_gas - 28.8032660577024 < tol
