import torch

ckpt = torch.load("models/best_frcnn.pt", map_location="cpu")
sd = ckpt["model"] if "model" in ckpt else ckpt

print("=== args ===")
print(ckpt.get("args", {}))

print("\n=== state_dict keys and shapes ===")
for k, v in sd.items():
    shape = tuple(v.shape) if hasattr(v, "shape") else v
    print(f"{k}: {shape}")
