import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# 🔹 Step 1: Load MobileNetV2 Backbone
class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(weights="IMAGENET1K_V1").features  # Updated syntax

    def forward(self, x):
        return self.mobilenet(x)  # Output: [1, 1280, 8, 8] for 256×256 input

# 🔹 Step 2: Adapter Layer (Conv Only — Upsample removed)
class Adapter(nn.Module):
    def __init__(self, in_channels=1280, out_channels=288):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.adapter(x)

# 🔹 Step 3: Pose Head (Landmark Regressor)
class PoseHead(nn.Module):
    def __init__(self, in_channels=288, num_landmarks=39, values_per_landmark=5):
        super().__init__()
        self.num_outputs = num_landmarks * values_per_landmark
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Conv2d(128, self.num_outputs, kernel_size=1)
        )

    def forward(self, x):
        x = self.regressor(x)       # → [1, 195, H, W]
        x = x.mean(dim=[2, 3])      # Global average pool → [1, 195]
        return x

# 🔹 Step 4: Full Model
class BlazePoseWithMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV2Backbone()
        self.adapter = Adapter()
        self.pose_head = PoseHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.adapter(x)
        x = self.pose_head(x)
        return x

# 🔹 Step 5: Export to ONNX
def export_to_onnx():
    model = BlazePoseWithMobileNet()
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)  # ← Your real input shape

    torch.onnx.export(
        model,
        dummy_input,
        "blazepose_mobilenetv2_1.0_stitched.onnx",
        input_names=["input"],
        output_names=["landmarks"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch_size"}, "landmarks": {0: "batch_size"}}
    )
    print("✅ Exported to blazepose_mobilenetv2_1.0_stitched.onnx")

if __name__ == "__main__":
    export_to_onnx()