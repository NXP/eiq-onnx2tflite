#
# Copyright 2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger as logger
from onnx2tflite.src.tflite_generator import tflite_model as tflite_model


def _buffer_has_data(t_buffer: tflite_model.Buffer) -> bool | None:
    """Determine if given buffer has any data in it."""
    # noinspection PyBroadException
    try:
        if t_buffer.data is None:
            return False

        size = t_buffer.data.size
        return size != 0

    except Exception as _: # noqa: BLE001 
        logger.d("'ModelBuilder.bufferHasData()' failed!")
        return None


def tensor_has_data(t_tensor: tflite_model.Tensor) -> bool:
    """Determine if given TFLite tensor has any data."""
    if t_tensor.tmp_buffer is None:
        return False

    res = _buffer_has_data(t_tensor.tmp_buffer)
    if res is None:
        res = False

    return res


def all_tensors_are_static(*list_of_tensors) -> bool:
    """Return True, if all tensors in 'list_of_tensors' have data stored in them.

    :param list_of_tensors: List of TFLite tensors to check.
    :return: True, if all tensors are static. False, if at least 1 is not static.
    """
    return all(tensor_has_data(t) for t in list_of_tensors)
