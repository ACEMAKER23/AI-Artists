import random
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImagePromptDataset(Dataset):
    def __init__(self, image_dir, prompts, size=512):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        self.prompts = prompts  # list of instance prompts

        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        prompt = random.choice(self.prompts)
        return image, prompt  # choose random prompt if multiple
