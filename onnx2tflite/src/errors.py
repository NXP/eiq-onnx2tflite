#
# Copyright 2023 Martin Pavella
# Copyright 2023, 2026 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""errors

Module defines error codes, the `ConversionError` exception and custom assertion helpers used
throughout the converter. Logging itself is done via the stdlib `logging` module; each module
should create its own logger with `logging.getLogger(__name__)`.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Code(Enum):
    """Error codes.

    The integer `value` is the machine-facing exit code returned by the CLI entrypoints.
    """

    INTERNAL_ERROR = 1
    GENERATED_MODEL_INVALID = 2
    INVALID_OPTIMIZATION = 3
    PREPROCESSING_ERROR = 4

    UNSUPPORTED_OPERATOR = 21
    UNSUPPORTED_ONNX_TYPE = 22
    UNSUPPORTED_OPERATOR_ATTRIBUTES = 23
    NOT_IMPLEMENTED = 24

    INVALID_TYPE = 31
    INVALID_TENSOR_SHAPE = 32
    INVALID_ONNX_OPERATOR = 33
    INVALID_ONNX_OPERATOR_ATTRIBUTE = 34
    INVALID_ONNX_MODEL = 35

    CONVERSION_IMPOSSIBLE = 41
    SHAPE_INFERENCE_ERROR = 42
    IO_PRESERVATION_ERROR = 43

    INVALID_INPUT = 51


class ConversionError(Exception):
    """Exception raised when the ONNX -> TFLite conversion cannot proceed.

    The exception carries a `Code` describing the error category. Its string representation
    prepends the category name, e.g. `[UNSUPPORTED_OPERATOR] - <msg>`. The integer
    `code.value` is used as the process exit code by the CLI entrypoints.
    """

    def __init__(self, code: Code, msg: str, cause: Exception | None = None):
        """Create a `ConversionError` with an error `code`, a message and an optional cause."""
        self.code = code

        self.msg = msg
        super().__init__(msg)
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return the human-readable error, prefixed with the category name."""
        return f"[{self.code.name}] - {self.msg}"



def expect_type(obj, expected_type, msg: str = "") -> None:
    """Log a warning if `obj` is not of `expected_type`. Does not raise."""
    if type(obj) is not expected_type:
        logger.warning(
            "%s:Object '%s' is of type '%s' where '%s' was expected!", msg, obj, type(obj), expected_type
        )


def require_type(obj, required_type, msg: str = "") -> None:
    """Raise a `ConversionError` if `obj` is not of `required_type`."""
    if type(obj) is not required_type:
        raise ConversionError(
            Code.INVALID_TYPE,
            msg + f":Object '{obj}' is of type '{type(obj)}' where '{required_type}' was required!",
        )


def internal_assert(truth_value: bool, msg: str = "") -> None:
    """Assert that `truth_value` is True. If not, raise a `ConversionError` with `INTERNAL_ERROR`.

    :param truth_value: Boolean to check.
    :param msg: Message to raise the error with.
    """
    if not truth_value:
        raise ConversionError(Code.INTERNAL_ERROR, msg)
