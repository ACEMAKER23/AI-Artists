from diffusers import StableDiffusionPipeline
from source_codes.config_loader import load_config
from source_codes.load_diffusion import load_stable_diffusion_pipeline
import torch
from source_codes.lora_creation import inject_lora_unet, load_lora_weights

cfg = load_config("configurations/all_configs.yaml")

pipe = load_stable_diffusion_pipeline(cfg)
'''
pipe.safety_checker = None
pipe.requires_safety_checker = False

pipe.unet = inject_lora_unet(pipe.unet, cfg["LORA_CFG"])
load_lora_weights(pipe.unet, "model_save/step_9500")
'''

image = pipe(
    "monet_style, a car",
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

image.save("output_images/final_test.png")