#
# Copyright 2023-2026 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import logging

import pytest

from onnx2tflite.src import errors
from onnx2tflite.src.errors import Code, ConversionError, expect_type, internal_assert, require_type


def test_conversion_error_str_uses_category_name():
    err = ConversionError(Code.UNSUPPORTED_OPERATOR, "some message")
    assert str(err) == "[UNSUPPORTED_OPERATOR] - some message"
    assert err.code is Code.UNSUPPORTED_OPERATOR
    assert err.msg == "some message"


def test_conversion_error_exit_code_is_code_value():
    err = ConversionError(Code.INVALID_INPUT, "bad input")
    assert err.code.value == 51


def test_conversion_error_chaining():
    cause = ValueError("original")
    err = ConversionError(Code.INTERNAL_ERROR, "wrapped", cause)
    assert err.__cause__ is cause


def test_internal_assert_raises_on_false():
    with pytest.raises(ConversionError) as e:
        internal_assert(False, "must hold")
    assert e.value.code is Code.INTERNAL_ERROR
    assert e.value.msg == "must hold"


def test_internal_assert_passes_on_true():
    internal_assert(True, "ok")  # Should not raise.


def test_require_type_raises_on_mismatch():
    with pytest.raises(ConversionError) as e:
        require_type(1, str)
    assert e.value.code is Code.INVALID_TYPE


def test_require_type_passes_on_match():
    require_type("hello", str)  # Should not raise.


def test_expect_type_logs_warning_without_raising(caplog):
    with caplog.at_level(logging.WARNING, logger=errors.logger.name):
        expect_type(1, str, "my_object")
    assert any("my_object" in record.message for record in caplog.records)


def test_logging_via_caplog(caplog):
    logger = logging.getLogger("onnx2tflite.test")
    with caplog.at_level(logging.INFO, logger="onnx2tflite.test"):
        logger.info("hello info")
        logger.warning("hello warning")

    messages = [record.message for record in caplog.records]
    assert "hello info" in messages
    assert "hello warning" in messages
