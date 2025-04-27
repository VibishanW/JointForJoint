import torch
import torchvision.models as models

# Load pretrained MobileNetV2_1.0
model = models.mobilenet_v2(pretrained=True).features
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input,
    "mobilenetv2_1.0.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=11
)

print("Exported mobilenetv2_1.0.onnx")
