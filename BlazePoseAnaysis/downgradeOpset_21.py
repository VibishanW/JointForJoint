import onnx
from onnx import version_converter

# Load original ONNX model
original_model = onnx.load("blazepose_mobilenetv2_1.0_stitched.onnx")

# Convert model to opset 21
converted_model = version_converter.convert_version(original_model, 21)

# Save the downgraded model
onnx.save(converted_model, "blazepose_opset21.onnx")

print("Model downgraded to opset 21 and saved as blazepose_opset21.onnx")
