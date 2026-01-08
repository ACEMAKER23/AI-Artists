#the following code would convert the
from PIL import Image
import os

train_dir_name = "../artist_images"
all_image_dir_name = "../all_images"
image_size = (512, 512)

os.makedirs(train_dir_name, exist_ok=True)

for files in os.listdir(all_image_dir_name):
    image = Image.open(os.path.join(all_image_dir_name, files)).convert("RGB")
    image = image.resize(image_size)  # IMPORTANT: reassign
    print("standardizing and saving image:", files)
    image.save(os.path.join(train_dir_name, "standardized_" + files))
