import torch
from diffusers import StableDiffusionPipeline

def load_stable_diffusion_pipeline(cfg):
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32
    }

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["base_model"],
        dtype=dtype_map[cfg["model"]["dtype"]],
        cache_dir=cfg["paths"].get("cache_dir", None) #this line is optional
    ).to(cfg["model"]["device"])

    return pipe


"""
unet = diffPipeline.unet
text_encoder = diffPipeline.text_encoder
vae = diffPipeline.vae
for param in vae.parameters():
    param.requires_grad = False

for param in text_encoder.parameters():
    param.requires_grad = False

for param in unet.parameters():
    param.requires_grad = False
"""

#prompt = "a cinematic oil painting of a castle on a cliff at sunset without a frame"
#image = diffPipeline(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
#image.save("test_output.png")
#print("Image saved as test_output.png")

