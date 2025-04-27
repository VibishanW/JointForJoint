import onnx

# Load the ONNX model
model = onnx.load("blazepose_mobilenetv2_1.0_stitched.onnx")

# Collect all known output and initializer names
defined_names = {init.name for init in model.graph.initializer}

for node in model.graph.node:
    for output in node.output:
        defined_names.add(output)

# Check for undefined inputs
broken_inputs = []

for node in model.graph.node:
    for input_name in node.input:
        if input_name and input_name not in defined_names:
            broken_inputs.append((node.name, input_name))

# Report results
if not broken_inputs:
    print("✅ No broken or undefined input nodes found.")
else:
    print("⚠️ Found undefined inputs:")
    for node_name, input_name in broken_inputs:
        print(f"  Node '{node_name}' has undefined input '{input_name}'")
