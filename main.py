from source_codes.config_loader import load_config
from source_codes.load_diffusion import load_stable_diffusion_pipeline
from source_codes.lora_creation import inject_lora_unet
from source_codes.image_class import ImagePromptDataset
from source_codes.train import LoRATrainer
from torch.utils.data import DataLoader

def main():
    cfg = load_config("configurations/all_configs.yaml")

    pipe = load_stable_diffusion_pipeline(cfg)

    # Freeze base model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Disable safety checker
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # Inject LoRA
    pipe.unet = inject_lora_unet(pipe.unet, cfg["LORA_CFG"])

    # Safety check
    trainable = [n for n, p in pipe.unet.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all("lora_" in n for n in trainable)
    print(f"Trainable params: {len(trainable)}")
    print("Sample:", trainable[:5])

    # Dataset
    dataset = ImagePromptDataset(
        cfg["paths"]["dataset_dir"],
        cfg["prompt"]["instance"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    trainer = LoRATrainer(pipe, dataloader, cfg)
    trainer.train()

if __name__ == "__main__":
    main()