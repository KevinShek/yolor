import torch
from thop import profile
from models.models import *

# Load Model
weights = ["../weights/pytorch/yolov4-640-hedrial.pt"]
cfg = "cfg/yolov4-640-heridal.cfg"
# model = attempt_load(weights, map_location="cpu")  # load FP32 model
model = Darknet(cfg).to("cpu")
try:
    ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
except:
    load_darknet_weights(model, weights[0])
# Dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Print Results
flops, params = profile(model, inputs=(dummy_input,))
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f}M")


