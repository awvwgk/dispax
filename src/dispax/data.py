"""
Atomic data
===========

Data arrays for atomic constants like covalent radii or van-der-Waals radii.
"""

import os.path as op
import jax.numpy as jnp

from . import constants
from .typing import Array


covalent_rad_2009 = constants.ANGSTROM_TO_BOHR * jnp.array(
    [
        *[0.00],  # None
        *[0.32, 0.46],  # H,He
        *[1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67],  # Li-Ne
        *[1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96],  # Na-Ar
        *[1.76, 1.54],  # K,Ca
        *[1.33, 1.22, 1.21, 1.10, 1.07],  # Sc-
        *[1.04, 1.00, 0.99, 1.01, 1.09],  # -Zn
        *[1.12, 1.09, 1.15, 1.10, 1.14, 1.17],  # Ga-Kr
        *[1.89, 1.67],  # Rb,Sr
        *[1.47, 1.39, 1.32, 1.24, 1.15],  # Y-
        *[1.13, 1.13, 1.08, 1.15, 1.23],  # -Cd
        *[1.28, 1.26, 1.26, 1.23, 1.32, 1.31],  # In-Xe
        *[2.09, 1.76],  # Cs,Ba
        *[1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51],  # La-Eu
        *[1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53],  # Gd-Yb
        *[1.46, 1.37, 1.31, 1.23, 1.18],  # Lu-
        *[1.16, 1.11, 1.12, 1.13, 1.32],  # -Hg
        *[1.30, 1.30, 1.36, 1.31, 1.38, 1.42],  # Tl-Rn
        *[2.01, 1.81],  # Fr,Ra
        *[1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49],  # Ac-Am
        *[1.49, 1.51, 1.51, 1.48, 1.50, 1.56, 1.58],  # Cm-No
        *[1.45, 1.41, 1.34, 1.29, 1.27],  # Lr-
        *[1.21, 1.16, 1.15, 1.09, 1.22],  # -Cn
        *[1.36, 1.43, 1.46, 1.58, 1.48, 1.57],  # Nh-Og
    ]
)
"""
Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197).
Values for metals decreased by 10 %.
"""


covalent_rad_d3 = 4.0 / 3.0 * covalent_rad_2009
"""D3 covalent radii used to construct the coordination number"""


def get_rc(numbers: Array) -> Array:
    """
    Get covalent radii for given atomic numbers.

    Parameters
    ----------
    numbers : Array
        Atomic numbers.

    Returns
    -------
    Array
        Covalent radii.
    """

    rcov = covalent_rad_d3[numbers]
    return rcov.reshape((-1, 1)) + rcov.reshape((1, -1))


r4_over_r2 = jnp.array(
    [
        *[0.0000],  # None
        *[8.0589, 3.4698],  # H,He
        *[29.0974, 14.8517, 11.8799, 7.8715, 5.5588, 4.7566, 3.8025, 3.1036],  # Li-Ne
        *[26.1552, 17.2304, 17.7210, 12.7442, 9.5361, 8.1652, 6.7463, 5.6004],  # Na-Ar
        *[29.2012, 22.3934],  # K,Ca
        *[19.0598, 16.8590, 15.4023, 12.5589, 13.4788],  # Sc-
        *[12.2309, 11.2809, 10.5569, 10.1428, 9.4907],  # -Zn
        *[13.4606, 10.8544, 8.9386, 8.1350, 7.1251, 6.1971],  # Ga-Kr
        *[30.0162, 24.4103],  # Rb,Sr
        *[20.3537, 17.4780, 13.5528, 11.8451, 11.0355],  # Y-
        *[10.1997, 9.5414, 9.0061, 8.6417, 8.9975],  # -Cd
        *[14.0834, 11.8333, 10.0179, 9.3844, 8.4110, 7.5152],  # In-Xe
        *[32.7622, 27.5708],  # Cs,Ba
        *[23.1671, 21.6003, 20.9615, 20.4562, 20.1010, 19.7475, 19.4828],  # La-Eu
        *[15.6013, 19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631],  # Gd-Yb
        *[14.5716, 15.8758, 13.8989, 12.4834, 11.4421],  # Lu-
        *[10.2671, 8.3549, 7.8496, 7.3278, 7.4820],  # -Hg
        *[13.5124, 11.6554, 10.0959, 9.7340, 8.8584, 8.0125],  # Tl-Rn
        *[29.8135, 26.3157],  # Fr,Ra
        *[19.1885, 15.8542, 16.1305, 15.6161, 15.1226, 16.1576, 14.6510],  # Ac-Am
        *[14.7178, 13.9108, 13.5623, 13.2326, 12.9189, 12.6133, 12.3142],  # Cm-No
        *[14.8326, 12.3771, 10.6378, 9.3638, 8.2297],  # Lr-
        *[7.5667, 6.9456, 6.3946, 5.9159, 5.4929],  # -Cn
        *[6.7286, 6.5144, 10.9169, 10.3600, 9.4723, 8.6641],  # Nh-Og
    ]
)
"""
PBE0/def2-QZVP atomic values calculated by S. Grimme in Gaussian (2010),
rare gases recalculated by J. Mewes with PBE0/aug-cc-pVQZ in Dirac (2018).
Also new super heavies Cn,Nh,Fl,Lv,Og and Am-Rg calculated at 4c-PBE/Dyall-AE4Z (Dirac 2022)
"""

sqrt_z_r4_over_r2 = jnp.sqrt(
    0.5 * (r4_over_r2 * jnp.sqrt(jnp.arange(r4_over_r2.shape[0])))
)


def get_qq(numbers: Array) -> Array:
    """
    Get scaling factor for C8 / C6 fraction.

    Parameters
    ----------
    numbers : Array
        Atomic numbers.

    Returns
    -------
    Array
        Scaling factor for C8 / C6 fraction.
    """
    r4r2 = sqrt_z_r4_over_r2[numbers]
    return 3 * r4r2.reshape(-1, 1) * r4r2.reshape(1, -1)


def _load_rvdw(dtype=jnp.float32) -> Array:
    return jnp.array(
        jnp.load(op.join(op.dirname(__file__), "data-rvdw.npy")),
        dtype=dtype,
    )


vdw_rad_d3 = _load_rvdw()


class Data:

    rc: Array

    qq: Array

    rvdw: Array

    __slots__ = ["rc", "qq", "rvdw"]

    def __init__(self, rc: Array, qq: Array, rvdw: Array):
        self.rc = rc
        self.qq = qq
        self.rvdw = rvdw

    @classmethod
    def from_numbers(cls, numbers: Array, dtype=jnp.float32) -> "Data":
        rc = get_rc(numbers)
        qq = get_qq(numbers)
        rvdw = vdw_rad_d3[numbers.reshape(-1, 1), numbers.reshape(1, -1)]
        return cls(rc, qq, rvdw)

    def __getitem__(self, item: Array) -> "Data":
        """
        Get a subset of the data.

        Parameters
        ----------
        item : Array
            Atomic numbers.

        Returns
        -------
        Data
            Subset of the data.
        """

        return self.__class__(
            self.rc[item.reshape(-1, 1), item.reshape(1, -1)],
            self.qq[item.reshape(-1, 1), item.reshape(1, -1)],
            self.rvdw[item.reshape(-1, 1), item.reshape(1, -1)],
        )
