#
# Copyright 2023 Martin Pavella
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""
    protobufFromJson

Simple script to convert an ONNX (Protocol Buffer) file to JSON format.
Takes one argument, which is the input JSON file.
"""

import sys

import onnx
from google.protobuf.json_format import MessageToJson

if len(sys.argv) != 2:
    print("Require 1 .onnx file!")
    exit(1)

file = sys.argv[1]

model = onnx.load(file)

json = MessageToJson(model)

# Shorten long lines
# json = [line if len(line) < 100 else line[0:50] for line in json.split("\n")]
# json = "\n".join(json)

print(json)
