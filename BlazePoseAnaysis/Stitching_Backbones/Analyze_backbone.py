import onnx

# Load the model
model = onnx.load("pose_landmark_full.onnx")

# Get model inputs
print("🔹 Inputs:")
for i in model.graph.input:
    shape = [dim.dim_value for dim in i.type.tensor_type.shape.dim]
    print(f"  {i.name}: {shape}")

# Get model outputs
print("\n🔹 Outputs:")
for o in model.graph.output:
    shape = [dim.dim_value for dim in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {shape}")

# Preview the first 10 nodes
print("\n🔹 First 10 nodes:")
for node in model.graph.node[:10]:
    print(f"  {node.op_type}: inputs={node.input}, outputs={node.output}")
