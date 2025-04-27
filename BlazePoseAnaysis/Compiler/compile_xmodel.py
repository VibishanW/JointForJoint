import subprocess
import os

# === CONFIG ===
input_onnx_path = "/workspace/blazepose_quant.onnx"
output_dir = "/workspace/compiled_model"
net_name = "blazepose_v2_aie"
arch_json = "/opt/vitis_ai/compiler/arch/aie/vck5000_aie.json"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# === Compile using subprocess ===
command = [
    "vai_c_xir",
    "--xmodel", input_onnx_path,
    "--arch", arch_json,
    "--output_dir", output_dir,
    "--net_name", net_name
]

print(f"Running command: {' '.join(command)}")
result = subprocess.run(command, capture_output=True, text=True)

# === Print output or error ===
if result.returncode == 0:
    print("✅ Compilation successful!")
    print(result.stdout)
else:
    print("❌ Compilation failed:")
    print(result.stderr)
