"""
Coordination number
===================

Evaluates a fractional coordination number for a given geometry or batch of geometries.

Example
-------
>>> numbers = jnp.array([7, 1, 1, 1])
>>> positions = jnp.array([
...     [+0.00000000000000, +0.00000000000000, -0.54524837997150],
...     [-0.88451840382282, +1.53203081565085, +0.18174945999050],
...     [-0.88451840382282, -1.53203081565085, +0.18174945999050],
...     [+1.76903680764564, +0.00000000000000, +0.18174945999050],
... ])
>>> displacement_fn, shift_fn = space.free()
>>> rcov = data.covalent_rad_d3[numbers]
>>> r0 = rcov.reshape((-1, 1)) + rcov.reshape((1, -1))
>>> cn_fn = exp_count_pair(displacement_fn, r0=r0)
>>> cn_fn(positions)
DeviceArray([2.9901006, 0.9977214, 0.9977214, 0.9977214], dtype=float32)
>>> numbers = jnp.array([8, 1, 1])
>>> positions = jnp.array([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ])
>>> displacement_fn, shift_fn = space.free()
>>> rcov = data.covalent_rad_d3[numbers]
>>> r0 = rcov.reshape((-1, 1)) + rcov.reshape((1, -1))
>>> cn_fn = exp_count_pair(displacement_fn, r0=r0)
>>> cn_fn(positions)
DeviceArray([1.9877865, 0.9947576, 0.9947576], dtype=float32)
"""

from __future__ import annotations

from functools import partial
import jax.numpy as jnp
from jax_md import energy, smap, space

from . import data, util
from .typing import Array, Callable


def exp_count(
    dr: Array,
    r0: Array,
    kcn: Array,
) -> Array:
    """
    Exponential counting function for computing coordination numbers.

    Parameters
    ----------
    dr : Array
        Displacement vector between two atoms.
    r0 : Array
        Sum of covalent radii of atoms.
    kcn : Array
        Steepness of exponential function.

    Returns
    -------
    Array
        Value of coordination count for the given displacement.
    """
    return 2 / (1 + jnp.exp(-kcn * (r0 / dr - 1)))


def exp_count_pair(
    displacement_or_metric: space.DisplacementOrMetricFn,
    r0: Array,
    kcn: Array = 16,
    r_onset: Array = 20.0,
    r_cutoff: Array = 25.0,
) -> Callable[[Array], Array]:
    """
    Convenience function to create a coordination number function.

    Parameters
    ----------
    displacement_or_metric : DisplacementOrMetricFn
        Function to compute the displacement between two atoms.
    r0 : Array
        Sum of covalent radii of atoms, either a scalar or a 2D array.
    kcn : Array
        Steepness of exponential function.
    r_onset : Array
        Onset for cutting off the counting function.
    r_cutoff : Array
        Cutoff for the counting function.

    Returns
    -------
    Callable[[Array], Array]
        Coordination number function.

    Notes
    -----
    Cannot define counting function with species since one axis is not reduced.
    """

    r0 = util.maybe_downcast(r0)
    kcn = util.maybe_downcast(kcn)

    return smap.pair(
        energy.multiplicative_isotropic_cutoff(exp_count, r_onset, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        species=None,
        r0=r0,
        kcn=kcn,
        reduce_axis=(1,),
    )
