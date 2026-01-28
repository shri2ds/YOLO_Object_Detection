import torch

# Load your YOLO model
model = torch.load('yolo_best.pt')
dummy_input = torch.randn(1, 3, 640, 640).cuda()

# Export to ONNX
torch.onnx.export(
    model, dummy_input, "yolo.onnx", 
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)