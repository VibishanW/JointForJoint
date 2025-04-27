import onnx
from onnx import helper, shape_inference

# === Load Models ===
mobilenet = onnx.load("mobilenetv2_1.0.onnx")
blazepose = onnx.load("pose_landmark_full.onnx")

# === Get MobileNet Info ===
mobilenet_output = mobilenet.graph.output[0].name
mobilenet_input = mobilenet.graph.input[0].name
mobilenet_nodes = mobilenet.graph.node
mobilenet_initializers = mobilenet.graph.initializer

# === DEBUG: Print all input names in BlazePose to help identify the correct connection point ===
print("🔍 Scanning BlazePose node inputs:")
inputs_found = set()
for node in blazepose.graph.node:
    for inp in node.input:
        inputs_found.add(inp)

sorted_inputs = sorted(list(inputs_found))
for inp in sorted_inputs:
    print(f" - {inp}")

# === Manually set the correct name once identified from above ===
# Replace this with the one that looks like it's coming out of Identity_2 visually in Netron
first_regression_node_input = "Identity_2_raw_output___7:0"

# === Extract regression head nodes from BlazePose ===
keep_nodes = []
copying = False
for node in blazepose.graph.node:
    if first_regression_node_input in node.input:
        copying = True
    if copying:
        keep_nodes.append(node)

# Safety check
if not keep_nodes:
    raise RuntimeError(f"Could not find regression head starting from input: {first_regression_node_input}")

# === Update input of first regression head node to match MobileNet output ===
keep_nodes[0].input[0] = mobilenet_output

# === Combine MobileNet backbone + BlazePose regression head ===
new_graph = helper.make_graph(
    list(mobilenet_nodes) + list(keep_nodes),
    "blazepose_with_mobilenetv2_1.0",
    [helper.make_tensor_value_info(mobilenet_input, onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
    blazepose.graph.output,
    initializer=list(mobilenet_initializers)
)

# === Build, infer shapes, and save ===
new_model = helper.make_model(new_graph)
new_model = shape_inference.infer_shapes(new_model)
onnx.save(new_model, "blazepose_mobilenetv2_1.0_stitched.onnx")

print("New model saved: blazepose_mobilenetv2_1.0_stitched.onnx")