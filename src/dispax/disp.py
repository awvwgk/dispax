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
>>> param = dict(s6=1.0, a1=0.49484001, s8=0.78981345, a2=5.73083694)  # r²SCAN-D3(BJ)
>>> ref = Reference.from_numbers(numbers)
>>> data = Data.from_numbers(numbers)
>>> displacement_fn, shift_fn = space.free()
>>> energy_fn = dispersion(displacement_fn, **param, data=data, ref=ref)
>>> energy_fn(positions0) - energy_fn(positions1)
DeviceArray(-0.00039642, dtype=float32)
"""

from functools import partial
import jax.numpy as jnp
from jax_md import energy, smap, space

from . import ncoord, model, util
from .data import Data
from .reference import Reference
from .typing import Array, Callable


def rational_damping(
    dr: Array,
    c6: Array,
    qq: Array,
    rvdw: Array,
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
    rvdw: Array
        Van der Waals radii for pairs of atoms.
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

    Example
    -------
    >>> numbers = jnp.array([6, 8, 7, 1, 1, 1])
    >>> positions = jnp.array([
    ...     [-0.55569743203406, +1.09030425468557, +0.00000000000000],
    ...     [+0.51473634678469, +3.15152550263611, +0.00000000000000],
    ...     [+0.59869690244446, -1.16861263789477, +0.00000000000000],
    ...     [-0.45355203669134, -2.74568780438064, +0.00000000000000],
    ...     [+2.52721209544999, -1.29200800956867, +0.00000000000000],
    ...     [-2.63139587595376, +0.96447869452240, +0.00000000000000],
    ... ])
    >>> param = dict(s6=1.0, a1=0.4535, s8=1.9435, a2=4.4752)  # TPSS-D3(BJ)
    >>> ref = Reference.from_numbers(numbers)
    >>> data = Data.from_numbers(numbers)
    >>> displ, shift = space.free()
    >>> energy_fn = dispersion(displ, **param, data=data, ref=ref, damping_fn=rational_damping)
    >>> energy_fn(positions)
    DeviceArray(-0.00311339, dtype=float32)
    """

    dr2 = dr * dr
    dr6 = dr2 * dr2 * dr2
    dr8 = dr6 * dr2

    rr = a1 * jnp.sqrt(qq) + a2
    rr2 = rr * rr
    rr6 = rr2 * rr2 * rr2
    rr8 = rr6 * rr2

    return -c6 * (s6 / (dr6 + rr6) + s8 * qq / (dr8 + rr8))


def zero_damping(
    dr: Array,
    c6: Array,
    qq: Array,
    rvdw: Array,
    s6: Array,
    s8: Array,
    rs6: Array,
    rs8: Array,
    alp6: Array,
) -> Array:
    """
    Zero damping function.

    Parameters
    ----------
    dr : Array
        Interatomic distance.
    c6 : Array
        Dispersion coefficients.
    qq : Array
        Fraction between C8 / C6 coefficient.
    rvdw: Array
        Van der Waals radii for pairs of atoms.
    s6 : Array
        Scaling parameter for dipole-dipole interaction.
    s8 : Array
        Scaling parameter for dipole-quadrupole interaction.
    rs6 : Array
        Range-separation parameter for dipole-dipole interaction.
    rs8 : Array
        Range-separation parameter for dipole-quadrupole interaction.
    alp6 : Array
        Alpha parameter for dipole-dipole interaction.

    Returns
    -------
    Array
        Dispersion energy.

    Example
    -------
    >>> numbers = jnp.array([6, 8, 8, 1, 1])
    >>> positions = jnp.array([
    ...     [-0.53424386915034, -0.55717948166537, +0.00000000000000],
    ...     [+0.21336223456096, +1.81136801357186, +0.00000000000000],
    ...     [+0.82345103924195, -2.42214694643037, +0.00000000000000],
    ...     [-2.59516465056138, -0.70672678063558, +0.00000000000000],
    ...     [+2.09259524590881, +1.87468519515944, +0.00000000000000],
    ... ])
    >>> param = dict(s6=1.0, rs6=1.166, s8=1.105, rs8=1.0, alp6=14.0)  # TPSS-D3(0)
    >>> ref = Reference.from_numbers(numbers)
    >>> data = Data.from_numbers(numbers)
    >>> displ, shift = space.free()
    >>> energy_fn = dispersion(displ, **param, data=data, ref=ref, damping_fn=zero_damping)
    >>> energy_fn(positions)
    DeviceArray(-0.00076195, dtype=float32)
    """

    dr2 = dr * dr
    dr6 = dr2 * dr2 * dr2
    dr8 = dr6 * dr2

    alp8 = alp6 + 2

    f6 = s6 / (1 + 6 * (rs6 * rvdw / dr) ** alp6)
    f8 = s8 / (1 + 6 * (rs8 * rvdw / dr) ** alp8)

    return -c6 * (f6 / dr6 + qq * f8 / dr8)


def dispersion(
    displacement_fn: space.DisplacementFn,
    data: Data,
    ref: Reference,
    cn_onset: Array = 20.0,
    cn_cutoff: Array = 25.0,
    e2_onset: Array = 55.0,
    e2_cutoff: Array = 60.0,
    damping_fn: Callable[[Array, ...], Array] = rational_damping,
    counting_fn: Callable[[Array, ...], Array] = ncoord.exp_count,
    **damping_param: dict[str, Array],
) -> Callable[[Array], Array]:
    """
    Create a dispersion energy function.

    Parameters
    ----------
    displacement_fn : space.DisplacementFn
        Displacement function.
    data : Data
        Atomic data for each atom.
    ref : Reference
        Dispersion coefficients for each atom.
    damping_fn : Callable[[Array, ...], Array]
        Damping function for computing the dispersion energy
    counting_fn : Callable[[Array, ...], Array]
        Counting function for computing the coordination number
    **damping_param : dict[str, Array]
        Parameters for the damping function.

    Returns
    -------
    Callable[[Array], Array]
        Dispersion energy function.
    """

    cn_fn = energy.multiplicative_isotropic_cutoff(counting_fn, cn_onset, cn_cutoff)
    e2_fn = energy.multiplicative_isotropic_cutoff(damping_fn, e2_onset, e2_cutoff)

    def compute_fn(positions, **kwargs):
        displ = partial(displacement_fn, **kwargs)
        dR = space.map_product(displ)(positions, positions)
        dr = space.distance(dR)

        mask = dr > 0

        cn = jnp.sum(jnp.where(mask, cn_fn(dr, rc=data.rc), 0), -1)
        weights = model.weight_references(cn, ref, model.gaussian_weight)
        c6 = model.atomic_c6(weights, ref)

        energy = jnp.where(
            mask,
            e2_fn(dr, c6=c6, qq=data.qq, rvdw=data.rvdw, **damping_param) / 2,
            0,
        )
        return jnp.sum(energy)

    return compute_fn
