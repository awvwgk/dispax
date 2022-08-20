"""
Utility functions for working with atomic numbers.
"""

from functools import partial
import jax.numpy as jnp


def real_atoms(numbers: jnp.ndarray) -> jnp.ndarray:
    """
    Return indices of real atoms.

    Parameters
    ----------
    numbers : ndarray
        Atomic numbers of atoms.

    Returns
    -------
    ndarray
        Indices of real atoms.

    Examples
    --------
    >>> numbers = jnp.array([1, 1, 1, 7, 0, 0])
    >>> real_atoms(numbers)
    DeviceArray([ True,  True,  True,  True, False, False], dtype=bool)
    """
    return numbers > 0


@partial(jnp.vectorize, signature="(n)->(n,n)")
def real_pairs(numbers: jnp.ndarray) -> jnp.ndarray:
    """
    Return indices of real pairs.

    Parameters
    ----------
    numbers : ndarray
        Atomic numbers of atoms.

    Returns
    -------
    ndarray
        Indices of real pairs.

    Examples
    --------
    >>> numbers = jnp.array([1, 1, 1, 7, 0, 0])
    >>> real_pairs(numbers)
    DeviceArray([[False,  True,  True,  True, False, False],
                 [ True, False,  True,  True, False, False],
                 [ True,  True, False,  True, False, False],
                 [ True,  True,  True, False, False, False],
                 [False, False, False, False, False, False],
                 [False, False, False, False, False, False]], dtype=bool)
    >>> numbers = jnp.array([[1, 1, 1, 7, 0, 0], [1, 1, 8, 1, 1, 8]])
    >>> real_pairs(numbers)
    DeviceArray([[[False,  True,  True,  True, False, False],
                  [ True, False,  True,  True, False, False],
                  [ True,  True, False,  True, False, False],
                  [ True,  True,  True, False, False, False],
                  [False, False, False, False, False, False],
                  [False, False, False, False, False, False]],
    <BLANKLINE>
                 [[False,  True,  True,  True,  True,  True],
                  [ True, False,  True,  True,  True,  True],
                  [ True,  True, False,  True,  True,  True],
                  [ True,  True,  True, False,  True,  True],
                  [ True,  True,  True,  True, False,  True],
                  [ True,  True,  True,  True,  True, False]]], dtype=bool)
    """
    real = real_atoms(numbers)
    return jnp.expand_dims(real, -2) & jnp.expand_dims(real, -1) & ~jnp.diag(real)


def maybe_downcast(x):
    if isinstance(x, jnp.ndarray) and x.dtype is jnp.dtype("float64"):
        return x
    return jnp.array(x, jnp.float32)
