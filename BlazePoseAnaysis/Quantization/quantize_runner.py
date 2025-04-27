from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnx
import numpy as np
import os

# === CONFIGURATION ===
input_model_path = "blazepose_mobilenetv2_1.0_stitched.onnx"
output_model_path = "blazepose_quant.onnx"
input_name = "input"
input_shape = (1, 3, 256, 256)
calib_batches = 10  # Number of fake batches for calibration

# === STEP 1: Fake Calibration Data Reader ===
class FakeDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            fake_input = {input_name: np.random.rand(*input_shape).astype(np.float32)}
            self.enum_data = iter([fake_input] * calib_batches)
        return next(self.enum_data, None)

# === STEP 2: Run Quantization ===
quantize_static(
    model_input=input_model_path,
    model_output=output_model_path,
    calibration_data_reader=FakeDataReader(),
    quant_format=QuantType.QUInt8,
    per_channel=False
)

print(f"\nQuantization complete! Saved to {output_model_path}")

