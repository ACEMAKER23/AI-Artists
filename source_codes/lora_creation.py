# source_codes/lora_inject.py
import torch
import torch.nn as nn
import os

class LoRALinear(nn.Module):
    def __init__(self, linear_module, rank, alpha):
        super().__init__()
        self.linear = linear_module
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Linear(linear_module.in_features, rank, bias=False).to(linear_module.weight.device)
        self.lora_B = nn.Linear(rank, linear_module.out_features, bias=False).to(linear_module.weight.device)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)

def inject_lora_unet(unet, lora_cfg):
    rank = lora_cfg["rank"]
    alpha = lora_cfg["alpha"]
    target_modules = lora_cfg["target_modules"]

    count = 0

    for name, module in list(unet.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            parent = unet
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)

            setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
            print(f"[LoRA] Injecting into: {name}")
            count += 1

    print(f"[LoRA] Total injected layers: {count}")
    return unet


def save_lora_weights(unet, save_dir):
    """
    Save only LoRA parameters in Diffusers-compatible format.
    """
    os.makedirs(save_dir, exist_ok=True)

    lora_state_dict = {
        k: v.cpu()
        for k, v in unet.state_dict().items()
        if "lora_" in k
    }

    torch.save(
        lora_state_dict,
        os.path.join(save_dir, "pytorch_lora_weights.bin")
    )

def load_lora_weights(unet, lora_dir):
    """
    Load LoRA weights into an already-injected UNet.
    """
    path = os.path.join(lora_dir, "pytorch_lora_weights.bin")
    state = torch.load(path, map_location="cpu")
    unet.load_state_dict(state, strict=False)