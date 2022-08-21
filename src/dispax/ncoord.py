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
>>> data = Data.from_numbers(numbers)
>>> cn_fn = smap.pair(
...     exp_count,
...     space.canonicalize_displacement_or_metric(displacement_fn),
...     rc=data.rc,
...     reduce_axis=(1,),
... )
>>> 2 * cn_fn(positions)
DeviceArray([2.9901006, 0.9977214, 0.9977214, 0.9977214], dtype=float32)
>>> dr = space.distance(space.map_product(displacement_fn)(positions, positions))
>>> cn = jnp.sum(jnp.where(dr > 0, exp_count(dr, rc=data.rc), 0), -1)
>>> cn
DeviceArray([2.9901006, 0.9977214, 0.9977214, 0.9977214], dtype=float32)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import smap, space

from .data import Data
from .typing import Array


def exp_count(
    dr: Array,
    rc: Array,
    kcn: Array = 16,
) -> Array:
    """
    Exponential counting function for computing coordination numbers.

    Parameters
    ----------
    dr : Array
        Displacement vector between two atoms.
    rc : Array
        Sum of covalent radii of atoms.
    kcn : Array
        Steepness of exponential function.

    Returns
    -------
    Array
        Value of coordination count for the given displacement.

    Example
    -------
    >>> numbers = jnp.array([8, 1, 1])
    >>> positions = jnp.array([
    ...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
    ...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
    ...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
    ... ])
    >>> displacement_fn, shift_fn = space.free()
    >>> data = Data.from_numbers(numbers)
    >>> dr = space.distance(space.map_product(displacement_fn)(positions, positions))
    >>> cn = jnp.sum(jnp.where(dr > 0, exp_count(dr, rc=data.rc), 0), -1)
    >>> cn
    DeviceArray([1.9877865, 0.9947576, 0.9947576], dtype=float32)
    """
    return 1 / (1 + jnp.exp(-kcn * (rc / dr - 1)))
