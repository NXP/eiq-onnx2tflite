#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    meta

Definitions of classes, that other classes in /src/onnx_parser/ inherit from.
"""

from typing import Iterable, Any, Callable

import onnx


class ONNXObject:
    """ Parent class of most objects in the /onnx_parser/ directory.
        Encapsulates the *Proto descriptor.
    """
    _descriptor: Any  # Type depends on particular onnx objects. Specified in child classes.

    def __init__(self, descriptor: Any) -> None:
        self._descriptor = descriptor
        self._init_attributes()

    def _init_attributes(self):
        """ Function is called from the constructor.
            Child classes should initialize their attributes from the '_descriptor' here!
        """
        pass


class ONNXOperatorAttributes:
    """ Parent class of every class in the '/onnx_parser/builtin_attributes/' directory.
        Represents an operator with its specific attributes.
    """

    """ Protobuf descriptor. Holds barely structured data, that represents the individual
        attributes of the operator. The data will be assigned to the subclasses attributes 
        for easier access. """
    _descriptor: Iterable[onnx.AttributeProto]

    def __init__(self, descriptor: Iterable[onnx.AttributeProto]) -> None:
        self._descriptor = descriptor
        self._default_values()
        self._init_attributes()

    def _default_values(self):
        """ Child class should assign default values to its attributes or 'None'
            if it doesn't have a default value.
        """
        pass

    def _init_attributes(self):
        """ Child class should initialize its attributes with values from the '_descriptor'. """
        pass


class ONNXScalarListAttribute(list[int | float]):
    """ Class represents an ONNX operator attribute, that is a list of scalar values and has 'name' and 'type'. """
    _descriptor: onnx.AttributeProto

    name: str
    type: onnx.AttributeProto.AttributeType

    # A callable (function), which takes this classes descriptor and returns its appropriate attribute.
    # For example for an ONNXIntListAttribute: 'lambda x: x.ints'.
    source: Callable[[onnx.AttributeProto], Iterable]

    def __init__(self, descriptor: onnx.AttributeProto, source: Callable[[onnx.AttributeProto], Iterable]):
        super().__init__()

        self._descriptor = descriptor
        self.source = source

        self.name = descriptor.name
        self.type = descriptor.type

        for item in source(descriptor):
            self.append(item)


class ONNXIntListAttribute(ONNXScalarListAttribute):
    """ List of ONNX integer attributes (ints). """

    def __init__(self, descriptor: onnx.AttributeProto):
        super().__init__(descriptor, lambda x: x.ints)


class ONNXFloatListAttribute(ONNXScalarListAttribute):
    """ List of ONNX float attributes (floats). """

    def __init__(self, descriptor: onnx.AttributeProto):
        super().__init__(descriptor, lambda x: x.floats)


class ONNXStringListAttribute(list[str]):
    """ List of ONNX string attributes (strings). """

    _descriptor: onnx.AttributeProto

    name: str
    type: onnx.AttributeProto.AttributeType

    def __init__(self, descriptor: onnx.AttributeProto):
        super().__init__()

        self._descriptor = descriptor

        self.name = descriptor.name
        self.type = descriptor.type

        for item in descriptor.strings:
            self.append(item.decode("UTF-8"))
