#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""types

Module implements functions that work with ONNX data types.
"""

import numpy as np
import onnx

from onnx2tflite.src import errors
from onnx2tflite.src.errors import ConversionError


def name_for_onnx_type(o_type: onnx.TensorProto.DataType) -> str:
    return onnx.TensorProto.DataType.Name(o_type)


def to_numpy_type(o_type: onnx.TensorProto.DataType) -> np.ScalarType:
    """Convert ONNX DataType to numpy dtype"""
    if o_type == onnx.TensorProto.UNDEFINED:
        raise ConversionError(errors.Code.CONVERSION_IMPOSSIBLE, "ONNX DataType 'UNDEFINED' is not supported.")

    if o_type == onnx.TensorProto.FLOAT:
        return np.dtype(np.float32)

    if o_type == onnx.TensorProto.UINT8:
        return np.dtype(np.uint8)

    if o_type == onnx.TensorProto.INT8:
        return np.dtype(np.int8)

    if o_type == onnx.TensorProto.UINT16:
        return np.dtype(np.uint16)

    if o_type == onnx.TensorProto.INT16:
        return np.dtype(np.int16)

    if o_type == onnx.TensorProto.INT32:
        return np.dtype(np.int32)

    if o_type == onnx.TensorProto.INT64:
        return np.dtype(np.int64)

    if o_type == onnx.TensorProto.STRING:
        return np.dtype(np.bytes_)

    if o_type == onnx.TensorProto.BOOL:
        return np.dtype(np.bool_)

    if o_type == onnx.TensorProto.FLOAT16:
        return np.dtype(np.float16)

    if o_type == onnx.TensorProto.DOUBLE:
        return np.dtype(np.float64)

    if o_type == onnx.TensorProto.UINT32:
        return np.dtype(np.uint32)

    if o_type == onnx.TensorProto.UINT64:
        return np.dtype(np.uint64)

    if o_type == onnx.TensorProto.COMPLEX64:
        return np.dtype(np.cdouble)

    if o_type == onnx.TensorProto.COMPLEX128:
        return np.dtype(np.clongdouble)

    if o_type == onnx.TensorProto.BFLOAT16:
        # numpy doesn't support bfloat16, and neither does TFLite. Perhaps convert to float32 if needed.
        raise ConversionError(errors.Code.NOT_IMPLEMENTED, "ONNX DataType 'BFLOAT16' is not yet supported.")

    raise ConversionError(errors.Code.CONVERSION_IMPOSSIBLE, f"ONNX DataType '{o_type}' is not yet supported.")
