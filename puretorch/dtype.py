import numpy as np

from enum import Enum

allowed_dtypes = ["f16", "f32", "f64", "int16", "int32", "int64", "long"]


class DType(Enum):
    """
    Enum for numpy dtype referencing.
    """

    f16 = np.float16
    f32 = np.float32
    f64 = np.float64
    int16 = np.int16
    int32 = np.int32
    int64 = np.int64
    long = np.int32


DTYPES = list(DType.__members__.keys())
