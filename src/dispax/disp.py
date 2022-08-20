"""
Dispersion energy
=================

This module provides the dispersion energy evaluation for the pairwise interactions.

Example
-------
>>> numbers = jnp.array([8, 1, 1, 8, 1, 6, 1, 1, 1])
>>> positions0 = jnp.array([
...     [-4.224363834, +0.270465696, +0.527578960],
...     [-5.011768887, +1.780116228, +1.143194385],
...     [-2.468758653, +0.479766200, +0.982905589],
...     [+1.146167671, +0.452771215, +1.257722311],
...     [+1.841554378, -0.628298322, +2.538065200],
...     [+2.024899840, -0.438480095, -1.127412563],
...     [+1.210773578, +0.791908575, -2.550591723],
...     [+4.077073644, -0.342495506, -1.267841745],
...     [+1.404422261, -2.365753991, -1.503620411],
... ])
>>> positions1 = jnp.concatenate(
...    (positions0[:3, :], positions0[3:, :] + jnp.array([100.0, 0.0, 0.0])), axis=0
... )
>>> c6 = jnp.array([
...     [10.4125013,  5.4365230,  5.4351273, 10.4113941,  5.4319568,
...      13.6274719,  5.4359927,  5.4365005,  5.4364958],
...     [ 5.4365230,  3.0927703,  3.0918815,  5.4358449,  3.0898616,
...       7.4717073,  3.0924325,  3.0927563,  3.0927527],
...     [ 5.4351273,  3.0918815,  3.0909929,  5.4344497,  3.0889738,
...       7.4696641,  3.0915439,  3.0918674,  3.0918639],
...     [10.4113941,  5.4358449,  5.4344497, 10.4102859,  5.4312797,
...      13.6258793,  5.4353147,  5.4358230,  5.4358177],
...     [ 5.4319568,  3.0898616,  3.0889740,  5.4312797,  3.0869563,
...       7.4650211,  3.0895240,  3.0898473,  3.0898442],
...     [13.6274719,  7.4717073,  7.4696641, 13.6258793,  7.4650211,
...      18.3402557,  7.4709311,  7.4716754,  7.4716673],
...     [ 5.4359927,  3.0924325,  3.0915437,  5.4353147,  3.0895243,
...       7.4709306,  3.0920947,  3.0924184,  3.0924149],
...     [ 5.4365005,  3.0927560,  3.0918674,  5.4358230,  3.0898476,
...       7.4716749,  3.0924184,  3.0927420,  3.0927389],
...     [ 5.4364958,  3.0927529,  3.0918639,  5.4358177,  3.0898442,
...       7.4716673,  3.0924149,  3.0927389,  3.0927355],
... ])
>>> param = dict(s6=1.0, a1=0.49484001, s8=0.78981345, a2=5.73083694)  # r²SCAN-D3(BJ)
>>> displacement_fn, shift_fn = space.free()
>>> r4r2 = data.sqrt_z_r4_over_r2[numbers]
>>> qq = 3 * r4r2.reshape(-1, 1) * r4r2.reshape(1, -1)
>>> energy_fn = rational_damping_pair(displacement_fn, c6=c6, qq=qq, **param)
>>> energy_fn(positions0) - energy_fn(positions1)
DeviceArray(-0.00039647, dtype=float32)
"""

import jax.numpy as jnp
from jax_md import energy, smap, space

from . import data, util
from .typing import Array, Callable


def rational_damping(
    dr: Array,
    c6: Array,
    qq: Array,
    s6: Array,
    s8: Array,
    a1: Array,
    a2: Array,
) -> Array:
    """
    Rational damping function.

    Parameters
    ----------
    dr : Array
        Interatomic distance.
    c6 : Array
        Dispersion coefficients.
    qq : Array
        Fraction between C8 / C6 coefficient.
    s6 : Array
        Scaling parameter for dipole-dipole interaction.
    s8 : Array
        Scaling parameter for dipole-quadrupole interaction.
    a1 : Array
        Scaling parameter for critical radius.
    a2 : Array
        Offset parameter for critical radius.

    Returns
    -------
    Array
        Dispersion energy.
    """

    dr2 = dr * dr
    dr6 = dr2 * dr2 * dr2
    dr8 = dr6 * dr2

    rr = a1 * jnp.sqrt(qq) + a2
    rr2 = rr * rr
    rr6 = rr2 * rr2 * rr2
    rr8 = rr6 * rr2

    return -c6 * (s6 / (dr6 + rr6) + s8 * qq / (dr8 + rr8))


def rational_damping_pair(
    displacement_or_metric: space.DisplacementOrMetricFn,
    c6: Array,
    qq: Array,
    s6: Array,
    s8: Array,
    a1: Array,
    a2: Array,
    r_onset: Array = 55.0,
    r_cutoff: Array = 60.0,
    per_particle: bool = False,
) -> Callable[[Array], Array]:
    """
    Convenience function to create a rational damping function.

    Parameters
    ----------
    displacement_or_metric : DisplacementOrMetricFn
        Function to compute the displacement between two atoms.
    c6 : Array
        Dispersion coefficients.
    qq : Array
        Fraction between C8 / C6 coefficient.
    s6 : Array
        Scaling parameter for dipole-dipole interaction.
    s8 : Array
        Scaling parameter for dipole-quadrupole interaction.
    a1 : Array
        Scaling parameter for critical radius.
    a2 : Array
        Offset parameter for critical radius.
    r_onset : Array
        Onset for cutting off the counting function.
    r_cutoff : Array
        Cutoff for the counting function.

    Returns
    -------
    Callable[[Array], Array]
        Coordination number function.
    """

    c6 = util.maybe_downcast(c6)
    qq = util.maybe_downcast(qq)
    s6 = util.maybe_downcast(s6)
    s8 = util.maybe_downcast(s8)
    a1 = util.maybe_downcast(a1)
    a2 = util.maybe_downcast(a2)

    return smap.pair(
        energy.multiplicative_isotropic_cutoff(rational_damping, r_onset, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        species=None,
        c6=c6,
        qq=qq,
        s6=s6,
        s8=s8,
        a1=a1,
        a2=a2,
        reduce_axis=(1,) if per_particle else None,
    )
