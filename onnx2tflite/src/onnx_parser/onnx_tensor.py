#
# Copyright 2023 Martin Pavella
# Copyright 2023 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    Module contains the internal representation of an onnx TensorProto and SparseProto.

    MUST be separate from the 'onnx_model.py' module, because of cyclical imports.
        'NodeProto' needs to import 'onnx_parser/builtin_attributes/Constant', which needs 'TensorProto' and
        'SparseProto'. If 'NodeProto' and 'TensorProto' are in the same module, we get an import cycle.
        So either all 3 are in the same module, or they are all in separate modules.

        'NodeProto' can be a part of 'onnx_model.py', but 'TensorProto' needs its own module.
"""

import typing

import numpy as np
import onnx

import onnx2tflite.src.logger as logger
import onnx2tflite.src.onnx_parser.meta.meta as meta
import onnx2tflite.src.onnx_parser.meta.types as types
from onnx2tflite.src import tensor_formatting


class Dimension(meta.ONNXObject):
    _descriptor: onnx.TensorShapeProto.Dimension  # Specify parent '_descriptor' type

    value: typing.Union[int, str]
    denotation: str

    def __init__(self, descriptor: onnx.TensorShapeProto.Dimension) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        self.denotation = self._descriptor.denotation

        if self._descriptor.HasField("dim_value"):
            self.value = self._descriptor.dim_value
        elif self._descriptor.HasField("dim_param"):
            self.value = self._descriptor.dim_param
        else:
            logger.w("ONNX TensorShape.Dimension has no valid value!")


class TensorShapeProto(meta.ONNXObject):
    dims: typing.List[Dimension]

    def __init__(self, descriptor: onnx.TensorShapeProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        self.dims = []
        for item in self._descriptor.dim:
            self.dims.append(Dimension(item))


class SparseTensor(meta.ONNXObject):
    _descriptor: onnx.TypeProto.SparseTensor  # Specify the exact type of '_descriptor' from parent class

    # The onnx schema for some reason specifies 'elem_type' as type int32. But comments in the schema state, that the
    # value of 'elem_type' MUST be a valid value of the 'TensorProto.DataType' enum.
    # Here, the type is 'TensorProto.DataType' in the first place, to reflect the comment.
    elem_type: onnx.TensorProto.DataType
    shape: TensorShapeProto

    def __init__(self, descriptor: onnx.SparseTensorProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        # noinspection PyTypeChecker
        self.elem_type = self._descriptor.elem_type  # Warning is probably OK. See 'elem_type' declaration above.
        self.shape = TensorShapeProto(self._descriptor.shape)


class Segment(meta.ONNXObject):
    _descriptor: onnx.TensorProto.Segment  # Specify parent '_descriptor' type

    begin: int
    end: int

    def __init__(self, descriptor: onnx.TensorProto.Segment) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        self.begin = self._descriptor.begin
        self.end = self._descriptor.end


def _has_data(field):
    """ Determine if given repeated field has data stored in it. """
    return (field is not None) and (len(field) != 0)


class TensorProto(meta.ONNXObject):
    dims: typing.List[int]
    data_type: onnx.TensorProto.DataType  # Schema defines as int32, but comment says value must be valid for the enum.
    # 'onnx-ml.proto' line '517'
    segment: Segment
    data: typing.Optional[np.ndarray]
    name: str
    doc_string: str
    # TODO externalData
    data_location: onnx.TensorProto.DataLocation

    tensor_format: tensor_formatting.TensorFormat

    def __init__(self, descriptor: onnx.TensorProto) -> None:
        super().__init__(descriptor)

    def _init_attributes(self):
        self.dims = list(self._descriptor.dims)
        self.data_type = self._descriptor.data_type  # Warning is probably fine. See 'dataType' declaration above.
        self.segment = Segment(self._descriptor.segment)
        self.name = self._descriptor.name
        self.doc_string = self._descriptor.doc_string
        self.data_location = self._descriptor.data_location

        self.tensor_format = tensor_formatting.TensorFormat.NONE

        self._assign_data()

        try:
            if self.data_type in [onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128]:
                # TODO Explicit support needs to be implemented. Probably not worth the time right now.
                logger.e(logger.Code.NOT_IMPLEMENTED, "Loading ONNX complex tensors is not yet supported!")

            # Some models use empty tensor with dimension [0]. Avoid reshaping.
            if self.dims != [0]:
                self.data = np.reshape(self.data, self.dims)
        except Exception as e:
            logger.e(logger.Code.INTERNAL_ERROR,
                     f"Could not reshape data of tensor '{self.name}' to shape '{self.dims}'", exception=e)

    def _assign_data(self):
        """ Assign tensor's data to the 'data' attribute from any initialized array in the descriptor.
             Also check that the data type used is allowed in the schema for the particular array.
        """

        self.data = None

        # Raw data
        if _has_data(self._descriptor.raw_data):
            # 'onnx-ml.proto' line '581'
            self._assert_type_not_banned([onnx.TensorProto.STRING, onnx.TensorProto.UNDEFINED], "raw_data")

            self.data = np.frombuffer(self._descriptor.raw_data, types.to_numpy_type(self.data_type))

        # Float data
        elif _has_data(self._descriptor.float_data):
            # 'onnx-ml.proto' line '540'
            self._assert_type_allowed([onnx.TensorProto.FLOAT, onnx.TensorProto.COMPLEX64], "float_data")

            if self.data_type == onnx.TensorProto.COMPLEX64:
                # TODO Explicit support needs to be implemented. Probably not worth the time right now.
                logger.e(logger.Code.NOT_IMPLEMENTED, "Loading ONNX complex tensors is not yet supported!")

            self.data = np.fromiter(self._descriptor.float_data, types.to_numpy_type(self.data_type))

        # Int32 data
        elif _has_data(self._descriptor.int32_data):
            # 'onnx-ml.proto' line '547'
            self._assert_type_allowed(
                [onnx.TensorProto.INT32, onnx.TensorProto.INT16, onnx.TensorProto.INT8, onnx.TensorProto.UINT16,
                 onnx.TensorProto.UINT8, onnx.TensorProto.BOOL, onnx.TensorProto.FLOAT16,
                 onnx.TensorProto.BFLOAT16], "int32_data")

            if self.data_type == onnx.TensorProto.FLOAT16:
                # int32_data is an array of int32. ONNX allows it to be used for float16, but numpy cannot correctly
                # load the float16 data directly from the array.
                data = bytes(np.fromiter(self._descriptor.int32_data, np.uint16))
                self.data = np.frombuffer(data, np.float16)

            else:
                self.data = np.fromiter(self._descriptor.int32_data, types.to_numpy_type(self.data_type))

        # String data
        elif _has_data(self._descriptor.string_data):
            logger.i("onnx_tensor.TensorProto._assignData(): string_data")

            # 'onnx-ml.proto' line '555'
            self._assert_type_allowed([onnx.TensorProto.STRING], "string_data")

            logger.e(logger.Code.NOT_IMPLEMENTED, "Loading ONNX string tensors is not yet implemented!")

        # Int64 data
        elif _has_data(self._descriptor.int64_data):
            # 'onnx-ml.proto' line '558'
            self._assert_type_allowed([onnx.TensorProto.INT64], "int64_data")

            self.data = np.fromiter(self._descriptor.int64_data, types.to_numpy_type(self.data_type))

        # Double data
        elif _has_data(self._descriptor.double_data):
            logger.i("onnx_tensor.TensorProto._assignData(): double_data")

            # 'onnx-ml.proto' line '612'
            self._assert_type_allowed([onnx.TensorProto.DOUBLE, onnx.TensorProto.COMPLEX128], "double_data")

            if self.data_type == onnx.TensorProto.COMPLEX128:
                # TODO Explicit support needs to be implemented. Probably not worth the time right now.
                logger.e(logger.Code.NOT_IMPLEMENTED, "Loading ONNX complex tensors is not yet supported!")

            self.data = np.fromiter(self._descriptor.double_data, types.to_numpy_type(self.data_type))

        # Uint64 data
        elif _has_data(self._descriptor.uint64_data):
            logger.i("onnx_tensor.TensorProto._assignData(): uint64_data")

            # 'onnx-ml.proto' line '617'
            self._assert_type_allowed([onnx.TensorProto.UINT32, onnx.TensorProto.UINT64], "uint64_data")

            if self.data_type == onnx.TensorProto.UINT32:
                # UINT32 and UINT64 have different size. Loadig may potentially be inaccurate.
                logger.i("onnx_tensor.TensorProto._assign_data(): uint64_data + UINT32.")

            self.data = np.fromiter(self._descriptor.uint64_data, types.to_numpy_type(self.data_type))

    def _assert_type_allowed(self, allowed_types: typing.List, for_field: str) -> bool:
        """ Check that 'self.dataType' is in 'allowedTypes'. If it isn't,
            print warning message.
            'allowedTypes' is a list of 'onnx.TensorProto.DataType' values.
            Return 'True' if type is allowed. """

        if self.data_type not in allowed_types:
            logger.w(f"ONNX Tensor '{for_field}' is used and 'data_type' is '{self.data_type}'! "
                     f"MUST be one of '{allowed_types}'.")
            return False

        return True

    def _assert_type_not_banned(self, banned_types: typing.List, for_field: str) -> bool:
        """ Check that 'self.dataType' is NOT in 'bannedTypes'. If it IS, print warning message.
            'bannedTypes' is a list of 'onnx.TensorProto.DataType' values.
            Return 'True' if type is not banned. """

        if self.data_type in banned_types:
            logger.w(f"ONNX Tensor '{for_field}' is used and 'data_type' is '{self.data_type}'! must NOT be "
                     f"one of '{banned_types}'.")
            return False

        return True


class RepeatedTensorProto(typing.List[TensorProto]):
    def __init__(self, descriptor_iterable: typing.MutableSequence[onnx.TensorProto]):
        super().__init__()
        for descriptor in descriptor_iterable:
            self.append(TensorProto(descriptor))
