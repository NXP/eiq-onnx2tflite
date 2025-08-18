# Using ONNXRT tags

Except of standard TODO and FIXME tags, additional tag is defined `ONNXRT` to mark identified limitation and differences
in the ONNX Runtime implementation compared to ONNX standard.

To use this tag in PyCharm, go to Settings -> Editor -> TODO:

- In the Patterns click on + to add new tag. Type `\bonnxrt\b.*` and keep the case sensitive unchecked.
- In the Filters click on + to add new filter. Name is e.g. "ONNX Runtime Limitations" and choose the afforementioned
  tag.

**Note:** To retrieve all the tags in PyCharm, go to View -> Tools Window -> TODO. To speed up the search by excluding
the search of
unrelated folder and projects, use the _Scope Based_ tab and define a new scope for the onnx2tflite - click on
the `...`.
Add a new scope with the + button and exclude other projects, venv/ and thirdparty/.