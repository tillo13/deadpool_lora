
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import re

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_mosaic(images, title, save_path):
    # Determine the size of the mosaic
    rows = -(-len(images) // 3)  # Ceiling division to determine number of rows
    width, height = images[0].size
    mosaic_image = Image.new('RGB', (3 * width, rows * height))

    # Paste images into mosaic
    for index, img in enumerate(images):
        x = (index % 3) * width
        y = (index // 3) * height
        mosaic_image.paste(img, (x, y))

    # Add title to mosaic
    draw = ImageDraw.Draw(mosaic_image)
    font = ImageFont.load_default()
    draw.text((10, 10), title, (255, 255, 255), font=font)

    # Save the mosaic image
    mosaic_image.save(save_path)

def create_mosaics():
    # Ensure the output directory exists
    output_dir = "./generated_images"
    mosaics_dir = "./mosaics"
    os.makedirs(mosaics_dir, exist_ok=True)

    # Define the schedulers
    schedulers = [
        "LMS", "DPM++_2M_Karras", "DDIM", "Euler_a", "Euler", "Heun",
        "DEIS", "DPM_Solver++", "KDPM2", "KDPM2_Ancestral", "UniPC", "PNDM"
    ]

    # Group images by scheduler
    scheduler_images = {scheduler: [] for scheduler in schedulers}
    
    for file_name in os.listdir(output_dir):
        match = re.match(r".*_(LMS|DPM\+\+_2M_Karras|DDIM|Euler_a|Euler|Heun|DEIS|DPM_Solver\+\+|KDPM2|KDPM2_Ancestral|UniPC|PNDM)_.*\.png", file_name)
        if match:
            scheduler = match.group(1)
            scheduler_images[scheduler].append(Image.open(os.path.join(output_dir, file_name)))

    # Create mosaics for each scheduler
    for scheduler, images in scheduler_images.items():
        if not images:
            continue

        # Create and save the mosaic for the current scheduler
        title = f"Scheduler: {scheduler}"
        mosaic_image_name = f"mosaic_{scheduler}_{get_timestamp()}.png"
        mosaic_image_path = os.path.join(mosaics_dir, mosaic_image_name)
        create_mosaic(images, title, mosaic_image_path)
        print(f"Created mosaic for scheduler {scheduler} saved as {mosaic_image_path}")

# Run the function to create mosaics
if __name__ == "__main__":
    create_mosaics()