#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

from onnx2tflite.src import logger
from onnx2tflite.src.logger import loggingContext

logger.MIN_OUTPUT_IMPORTANCE = logger.MessageImportance.INFO


def test_nested_logging_context():
    logger.w("first")  # ignored
    with loggingContext(logger.BasicLoggingContext.ONNX_PARSER):
        logger.i("second")
        with loggingContext(logger.NodeLoggingContext(10)):
            try:
                logger.e(logger.Code.INTERNAL_ERROR, "third")
            except Exception:
                pass
    logger.w("fourth")  # ignored

    log = logger.conversion_log.get_logs()

    assert len(log.keys()) == 2
    assert [data[0]["message"] for data in log.values()] == ["second", "[Code.INTERNAL_ERROR] - third"]
    assert [data[0]["importance"] for data in log.values()] == [1, 3]

    logger.conversion_log.reset()


def test_log_no_context():
    logger.i("parser_log")

    log = logger.conversion_log.get_logs()

    assert len(log.keys()) == 0

def test_reset_after_reentering_root_context():
    with loggingContext(logger.BasicLoggingContext.ONNX_PARSER):
        logger.i("first")

    with loggingContext(logger.BasicLoggingContext.ONNX_PARSER):
        logger.i("second")

    log = logger.conversion_log.get_logs()

    assert len(log.keys()) == 1
    assert log["onnx_parser"][0]["message"] == "second"

    logger.conversion_log.reset()
