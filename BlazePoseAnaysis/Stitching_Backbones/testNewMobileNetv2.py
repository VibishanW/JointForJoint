import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("blazepose_mobilenetv2_1.0_stitched.onnx")
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

outputs = sess.run(None, {"input": dummy_input})
print("Output shape:", outputs[0].shape)  # Expect (1, 195)
