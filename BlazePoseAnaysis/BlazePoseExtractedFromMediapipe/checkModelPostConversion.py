import onnx

onnx_model = onnx.load("pose_landmark_full.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model is valid!")
