"""
Example
-------
>>> numbers = jnp.array([8, 1, 1])
>>> positions = jnp.array([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ])
>>> ref = Reference.from_numbers(numbers)
>>> cn = jnp.array([1.9877865, 0.9947576, 0.9947576])
>>> weights = weight_references(cn, ref, gaussian_weight)
>>> jnp.allclose(jnp.sum(weights, -1), 1)
DeviceArray(True, dtype=bool)
>>> c6 = atomic_c6(weights, ref)
>>> c6
DeviceArray([[10.413044 ,  5.4368806,  5.4368806],
             [ 5.4368806,  3.0930142,  3.0930142],
             [ 5.4368806,  3.0930142,  3.0930142]], dtype=float32)
"""

import jax.numpy as jnp

from .reference import Reference
from .typing import Array, Callable


def atomic_c6(weights: Array, reference: Reference) -> Array:
    """
    Compute atomic dispersion coefficients from reference weights.

    Parameters
    ----------
    weights : Array
        Weights for each atom and reference.
    reference : Reference
        Reference dispersion coefficients.

    Returns
    -------
    Array
        Atomic dispersion coefficients.
    """

    gw = (
        weights[jnp.newaxis, :, jnp.newaxis, :]
        * weights[:, jnp.newaxis, :, jnp.newaxis]
    )
    return jnp.sum(gw * reference.c6, axis=(-1, -2))


def gaussian_weight(dcn: Array, factor: Array = 4) -> Array:
    """
    Calculate weight of indivdual reference system.

    Parameters
    ----------
    dcn : Array
        Difference of coordination numbers.
    factor : Array
        Factor to calculate weight.

    Returns
    -------
    Array
        Weight of individual reference system.
    """

    return jnp.exp(-factor * dcn**2)


def weight_references(
    cn: Array,
    reference: Reference,
    weighting_function: Callable[[Array, ...], Array] = gaussian_weight,
    **kwargs,
) -> Array:
    """
    Calculate the weights of the reference system.

    Parameters
    ----------
    cn : Array
        Coordination numbers for all atoms in the system.
    reference : Reference
        Reference systems for D3 model.
    weighting_function : Callable
        Function to calculate weight of individual reference systems.

    Returns
    -------
    Array
        Weights of all reference systems
    """

    weights = jnp.where(
        reference.cn >= 0,
        weighting_function(reference.cn - cn.reshape((-1, 1)), **kwargs),
        0,
    )
    norms = jnp.sum(weights, -1) + jnp.finfo(cn.dtype).eps

    return weights / norms.reshape((-1, 1))
