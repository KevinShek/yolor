import torch
from thop import profile
from models.experimental import attempt_load

# Load Model
weights = "../weights/pytorch/yolov4-csp-640-hedrial.pt"
model = attempt_load(weights, map_location="cpu")  # load FP32 model

# Dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Print Results
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f}M")


