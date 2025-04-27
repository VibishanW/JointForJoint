import tensorflow as tf
import tf2onnx
import onnx

# Load the TFLite model
tflite_model_path = "pose_landmark_full.tflite"
onnx_model_path = "pose_landmark_full.onnx"

# Load the TFLite model as bytes
with open(tflite_model_path, "rb") as f:
    tflite_model = f.read()

# Convert TFLite model to TensorFlow model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_tflite(
    tflite_model_path, 
    output_path=onnx_model_path, 
    opset=13
)

# Check ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

print("Converted BlazePose TFLite model to ONNX successfully!")
