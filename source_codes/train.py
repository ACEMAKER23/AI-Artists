import os
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm
from source_codes.lora_creation import save_lora_weights

class LoRATrainer:
    def __init__(self,pipe,dataloader,cfg):
        self.pipe = pipe
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = pipe.device

        # Noise scheduler (same config as inference scheduler)
        self.noise_scheduler = DDPMScheduler.from_config(
            pipe.scheduler.config
        )

        print(cfg["training"]["learning_rate"], type(cfg["training"]["learning_rate"]))

        # Optimizer (LoRA params only)
        lora_params = [p for p in self.pipe.unet.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=float(cfg["training"]["learning_rate"]),  # initial LR
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["training"]["max_steps"],  # total steps
            eta_min=1e-6  # final LR
        )

        self.global_step = 0
        self.output_dir = cfg["paths"]["model_save_dir"]
        self.output_image_dir = cfg["paths"]["image_out_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        max_steps = self.cfg["training"]["max_steps"]
        save_every = self.cfg["training"]["save_every"]
        grad_accum = self.cfg["training"].get(
            "gradient_accumulation_steps", 1
        )

        progress_bar = tqdm(total=max_steps)

        self.pipe.unet.train()

        while self.global_step < max_steps:
            for images, prompts in self.dataloader:
                loss = self.training_step(images, prompts)
                loss = loss / grad_accum
                loss.backward()

                if (self.global_step + 1) % grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                if self.global_step % 500 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Step {self.global_step} | LR = {lr:.6f}")

                if self.global_step % 50 == 0:
                    progress_bar.set_postfix(loss=loss.item())

                if self.global_step % save_every == 0 and self.global_step > 0:
                    self.save_checkpoint()

                if self.global_step == 500:
                    print("500 steps")
                if self.global_step > 1 and self.global_step % self.cfg["training"]["sample_every"] == 0:
                    self.generate_sample()

                progress_bar.update(1)
                self.global_step += 1

                if self.global_step >= max_steps:
                    break

        progress_bar.close()
        self.save_checkpoint(final=True)

    def training_step(self, images, prompts):
        images = images.to(self.device)

        # 1. Encode images to latent space
        latents = self.pipe.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        # 2. Sample noise
        noise = torch.randn_like(latents)
        '''
        # 3. Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device
        ).long()
        '''
        timesteps = torch.randint(
            low=200,
            high=800,
            size=(latents.shape[0],),
            device=self.device
        )

        # 4. Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # 5. Encode text prompts
        text_inputs = self.pipe.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        text_embeddings = self.pipe.text_encoder(
            text_inputs.input_ids
        )[0]

        # 6. UNet predicts noise
        noise_pred = self.pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample

        # 7. Compute loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def save_checkpoint(self, final=False):
        name = "final" if final else f"step_{self.global_step}"
        save_path = os.path.join(self.output_dir, name)

        save_lora_weights(self.pipe.unet, save_path)

        print(f"[LoRA] Saved checkpoint to {save_path}")

    def generate_sample(self):
        self.pipe.unet.eval()

        with torch.no_grad():
            image = self.pipe(
                self.cfg["sampling"]["prompt"],
                num_inference_steps=self.cfg["sampling"]["num_inference_steps"],
                guidance_scale=self.cfg["sampling"]["guidance_scale"]
            ).images[0]

        save_path = os.path.join(
            self.output_image_dir,
            f"sample_step_{self.global_step}.png"
        )
        image.save(save_path)

        self.pipe.unet.train()
