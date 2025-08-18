#
# Copyright 2023-2024 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#
import logging
import os
import pathlib

import onnx
import onnxruntime
import pytest
import tensorflow as tf

import onnx2tflite.lib.tflite.BuiltinOperator as libBuiltinOperator
from onnx2tflite.src import logger
from onnx2tflite.src.conversion_config import ConversionConfig
from onnx2tflite.src.converter.builder import model_builder
from onnx2tflite.src.tflite_generator import tflite_model

logger.MIN_OUTPUT_IMPORTANCE = logger.MessageImportance.DEBUG
pylogger = logging.getLogger("onnx2tflite")
pylogger.setLevel(logging.DEBUG)

@pytest.fixture
def artifacts_dir():
    return pathlib.Path(__file__).parent.joinpath("artifacts")


@pytest.fixture
def cifar10_tflite(artifacts_dir):
    return {
        "model_path": os.path.join(artifacts_dir, "models", "cifar10_model.tflite"),
        "inputs_dir": os.path.join(artifacts_dir, "inputs", "image-32x32")
    }


# noinspection SpellCheckingInspection
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

    try:
        from pytest_metadata.plugin import metadata_key
        config.stash[metadata_key]["TensorFlow"] = tf.__version__
        config.stash[metadata_key]["ONNX"] = onnx.__version__
        config.stash[metadata_key]["ONNX Runtime"] = onnxruntime.__version__
    except ModuleNotFoundError:
        # PyTest Metadata not installed. Skipping.
        pass


# noinspection SpellCheckingInspection
def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class IntermediateTFLiteModelProvider:
    mocked_builder: model_builder.ModelBuilder

    def __init__(self, mocker):
        """ Create a single `ModelBuilder` object, which will be returned every time the `ModelBuilder()` constructor
             is called. This class holds a reference to this singleton `ModelBuilder`, which will be used during
             conversion in the tests. This strategy provides access to the converted TFLite model in the internal
             format, via `mocked_builder`.
        """
        self.mocked_builder = model_builder.ModelBuilder(3, "mocked_builder")
        mocker.patch("onnx2tflite.src.converter.builder.model_builder.ModelBuilder", side_effect=self._get_builder)

    def _get_builder(self, *args):
        if len(args) >= 3:
            # The 3rd argument should be a `ConversionConfig`.
            assert isinstance(args[2], ConversionConfig)
            self.mocked_builder.conversion_config = args[2]

        return self.mocked_builder

    def get(self) -> tflite_model.Model:
        return self.mocked_builder._tfl_model

    def get_last_graph(self) -> tflite_model.SubGraph:
        return self.get().sub_graphs.get_last()

    def get_operators(self) -> list[tflite_model.Operator]:
        return self.get_last_graph().operators.vector

    def get_tensors(self) -> list[tflite_model.Tensor]:
        return self.get_last_graph().tensors.vector

    def get_buffers(self) -> list[tflite_model.Buffer]:
        return self.mocked_builder.get_buffers().vector

    def get_operator_code_at_index(self, idx) -> libBuiltinOperator.BuiltinOperator:
        return self.get().operator_codes.get(idx).builtin_code

    def get_op_count(self, builtin_options) -> int:
        """
        Get count of operator occurrence in the last graph of the model.

        :param builtin_options: Builtin options representing searched operator.
        :return: Count of operator occurrence in last graph.
        """
        ops = filter(lambda x: isinstance(x.builtin_options, builtin_options), self.get_operators())
        return len(list(ops))

    def assert_converted_model_has_operators(self, expected_ops: list[libBuiltinOperator.BuiltinOperator]):
        """ Assert that the converted model contain operators listed in `expected_ops` and in that specific order.

        :param expected_ops: BuiltinOperator values which should be in the converted model, in the specific order.
        """
        ops = self.get_operators()

        assert len(ops) == len(expected_ops)

        for op, expected_opcode in zip(ops, expected_ops):
            opcode = self.get_operator_code_at_index(op.opcode_index)
            assert opcode == expected_opcode


@pytest.fixture
def intermediate_tflite_model_provider(mocker) -> IntermediateTFLiteModelProvider:
    return IntermediateTFLiteModelProvider(mocker)
