## New operator support

This guide defines a non-exhaustive list of topics that should be considered when implementing support for the new
operator into the ONNX to TFLite converter.

Whenever we're not able to convert the operator into its TFLite representation, we have to end conversion gracefully
via ``logger.e()`` to avoid errors during the inference time.

### Operator conversion integration

The operator conversion process starts with ONNX operator parsing. This is achieved by implementing attributes parser
named `<onnx_operator_name>_attributes.py` in
directory [src/onnx_parser/builtin_attributes](../onnx2tflite/src/onnx_parser/builtin_attributes) and registering such a
parser in
method `NodeProto::__assign_particular_operator_attributes()` ([onnx_model.py](../onnx2tflite/src/onnx_parser/onnx_model.py)).
Some operators don't contain any attributes (`QLinearAdd`, `Equal` etc.) so registration could assign value `None` as
operator attributes.

During the second phase is internal ONNX operator representation converted into the internal TFLite operator
representation. Similar to the previous phase, such an operator converter has to be added to the
directory [src/converter/node_converters](../onnx2tflite/src/converter/node_converters) following name
convention `<onnx_operator_name>_converter.py` and registered within the
method `convert_operator()` ([operator_converter.py](../onnx2tflite/src/converter/conversion/operator_converter.py)).
The operator converter file should contain a class which inherits from
the [NodeConverter](../onnx2tflite/src/converter/node_converter.py):

```python
from onnx2tflite.src.converter.node_converter import NodeConverter
from onnx2tflite.src.onnx_parser import onnx_model
from onnx2tflite.src.tflite_generator import tflite_model


class YourOperatorConverter(NodeConverter):
    node = 'YourOperator'

    onnx_supported_types = []  # List of data types supported by the ONNX documentation.
    tflite_supported_types = []  # List of data types supported by the TFLite inference engine.

    # List of data types which are convertible. This is usually the intersection of the previous 2 lists, minus types 
    #  unsupported by ONNX Runtime. There should always be a unit test which tests all of these types.
    verified_types = []

    def convert(self, node: onnx_model.NodeProto, t_op: tflite_model.Operator, ) -> list[tflite_model.Operator]:
        """
            Convert ONNX operator `YourOperator` into TFLite operator(s) ... .
            
            :param node: ONNX `YourOperator` operator.
            :param t_op: Empty TFLite operator with IO tensors in NHWC format, which correspond to the IO tensors of the 
                          ONNX operator.
            :return: List of TFLite operators which result from the conversion. They should represent equal behavior as 
                      the original ONNX operator. 
        """
        pass
```

### Input/output tensors

Input and output tensors of an operator might be accessed through list properties `t_op.tmp_inputs`
and `t_op.tmp_outputs` respectively. Firstly, it is necessary to validate if both ONNX and TFLite support the same set
of tensor data types and prohibit conversion of unsupported ones. TFLite seems more strict about input/output data type
combinations and some variants are not supported (input type not same as output type etc.). If we skip this step, we
might end up with a model that will fail during inference time.

Many ONNX operators declare some input tensors as optional in the documentation. ONNX allows multiple ways to
represent omitted inputs. The best way to access optional input tensors during conversion is via the
[common.py::try_get_input()](../onnx2tflite/src/converter/conversion/common.py) function.

If we need to change some aspect of input/output tensors, we have two options based on the tensor type - static
ones (tensor has data) or dynamic ones (computed during inference time). We can use
method [model_builder.py::tensor_has_data()](../onnx2tflite/src/converter/builder/model_builder.py) to
distinguish between the two. Static tensors can be manipulated directly - we can modify the shape, transpose data,
change the type, re-quantize data, etc. Dynamic tensors have to be prepended/appended with other TFLite operators to
achieve the desired behavior. [model_builder.py](../onnx2tflite/src/converter/builder/model_builder.py) contains plenty
of predefined utility functions like `create_transpose_operator_before()` or `create_quantize_operator_after()` that
could be used to manipulate input/output tensor data.

Sometimes, new tensors have to be introduced to satisfy TFLite inference engine requirements, for example, adding
missing bias tensor that is not mandatory in ONNX. There are multiple ways to construct a new tensor - duplicating
existing one ([model_builder::duplicate_tensors()](../onnx2tflite/src/converter/builder/model_builder.py)) or creating
brand new
one ([model_builder::create_tensor_for_data()](../onnx2tflite/src/converter/builder/model_builder.py), [model_builder::create_empty_tensor()](../onnx2tflite/src/converter/builder/model_builder.py),
etc.).

### Tensor format

ONNX tensors follow the format NCHW (batch|channels|height|width) whereas TFLite works in NHWC format (for 4D tensors).
Because of that, we run the tensor inference phase, directly after model parsing. During this phase, every tensor is,
based on its type and surrounding operators, assigned one of the formats - `FORMAT_LESS` or `CHANNELS_FIRST`.

Right before the operator conversion, the TFLite operator wrapper `t_op` is created and tensor formats and shapes are
changed to ones reflecting the TFLite world - `FORMAT_LESS` or `CHANNELS_LAST`. This changes the shapes of the tensors
and it permutes their static data to match. For dynamic tensors, only the shape and tensor format is changed.

Method `TensorFormat.is_channels_last()` can be used to find out if we're dealing with the channel-last tensor and act
accordingly. Just to give an intuition when it is handy, consider for example `axis` attribute of 'Concat' operator.
When we're dealing with `FORMAT_LESS` tensors, we can take `axis` as is and use the same value in TFLite. Situation
changes for `CHANNEL_LAST` input tensor because the order of dimensions has changed from, let's say `(1, 3, 48, 48)` (
ONNX) into `(1, 48, 48, 3)` (TFLite). Potential value 'axis=1' no longer corresponds to the number of channels 'C' but
to the height 'H'. This can be solved with axis recalculation.

Operators with the ability to change the tensor format (`Reshape`, `Squeeze`, etc.) should be given special care because
we have to take this information into account during the tensor inference phase. Such operators should be mentioned
in [tensor_format_inference.py](../onnx2tflite/src/tensor_format_inference.py) and assigned proper format. Every
operator which can have a different input and output rank should be included here. The conversion modules for these
operators must support all combinations of tensor formats for their inputs and outputs. This is usually done by
prepending or appending `Transpose` operators to make the inputs/outputs channels-first, but sometimes it is possible to
find more clever solutions. For inspiration, take a look at
[transpose_converter.py](../onnx2tflite/src/converter/node_converters/transpose_converter.py), or any module converting
one of these operators.

### Broadcasting

Broadcasting allows us to perform arithmetic operations on arrays with different shapes or ranks. Generally, it is
supported by both ONNX and TFLite, but the situation gets complicated due to NCHW/NHCW conversion (array dimensions are
mixed and thus no longer broadcastable between each other). This can be solved using the function
[model_builder::ensure_correct_broadcasting()](../onnx2tflite/src/converter/builder/model_builder.py) that
prepend/manipulate input tensor via transposition.

### Quantization

ONNX and TFLite use different approaches to quantization. In ONNX, quantization is defined on the operator level (
operator inputs contain scales and zero point tensors) and in TFLite on the tensor level (scales and zp assigned
directly to the tensor). We can assign quantization parameters to the tensor with
function [model_builder::set_quantization_parameters_to_tensor()](../onnx2tflite/src/converter/builder/model_builder.py).
Dynamic quantization (scales/zp computed in inference time) is not supported.

'Convert' function also has to make sure quantization parameters are properly propagated from input tensor to output.
Most of the TFLite operators don't support quantization of just input or just output, so we would end up in inference
time exception. Quantization parameters could be easily propagated between the tensors
with [model_builder::propagate_quantization()](../onnx2tflite/src/converter/builder/model_builder.py).

### Operator skipping

Some ONNX operators behave as identity/no-op (`Dropout`, specific `Transpose` case, etc.) and can be skipped during the
model conversion. In such cases we have to skip the operator and re-wire the input tensor directly to the output. We
also have to keep in mind that such an operator might be the only operator in the model. This can be solved with the
following code:

```python
if builder.operator_can_be_skipped(t_op, self.inspector):
    builder.redirect_tensor(t_op.tmp_outputs[0], t_op.tmp_inputs[0])
    return []
else:
    # The operator consumes a graph input tensor and also produces a graph output tensor.
    # We can return a Transpose op, which does nothing.
    builder.turn_operator_to_identity(t_op)
    return [t_op]
```

### Flex (Select) operators

Adding Flex operators (`SELECT_TF_OPS`) is not as straightforward as adding operator from TFLite builtin set. Example
implementation of an Erf Flex operator can be found in
[erf_converter.py](../onnx2tflite/src/converter/node_converters/erf_converter.py). Flex operators are defined by its
name and `custom_options`. `custom_options` define operator metadata in FlexBuffer format and are generated during the
conversion to TFLite. Follow these steps to obtain options for your operator:

1. Create single layer TFLite model with requested operator:

```python
import keras
import tensorflow as tf
from keras import layers
from tensorflow.lite.python.lite import TFLiteConverter

model_name = "erf_model"

inputs = tf.keras.Input(shape=(3, 3, 5), dtype=tf.float32)
outputs = layers.Lambda(lambda x: tf.math.erf(x=x), name="erf_lambda")(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile()
model.save(model_name)

converter = TFLiteConverter.from_saved_model(model_name)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.optimizations = []
tflite_model = converter.convert()

open(model_name + ".tflite", "wb").write(tflite_model)
```

2. Convert created TFLite model into its JSON representation by running (from project root):

```commandline
TFL=erf_model.tflite make tflite-to-json
```

3. Find corresponding `custom_options` ('root > subgraphs > operators') in generated JSON file and assign them
   to `t_op.custom_options` within implemented `convert()` function. `custom_options` is defined as a list of integers.
