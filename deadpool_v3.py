import os
import time
import torch
import random
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers import (
    LMSDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler,
    EulerDiscreteScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler, DEISMultistepScheduler, DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler
)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
# Define 10 action scenes with Deadpool
action_scenes = [
    "Deadpool leaping through flames, rescuing a child, intense facial expression, fire roaring in the background, dramatic lighting, high detail, action-packed scene, shattered glass flying, superhero in action.",
    "Deadpool in combat with a group of ninjas, dark alley setting, motion blur on weapons, dynamic poses, sweat and determination on faces, shadows and moonlight, detailed ninja outfits, swift and fluid movement, sparks from clashing swords, high-action choreography.",
    "Deadpool in mid-air, jumping off a skyscraper, dual-wielding pistols, trailing bullets, adrenaline-fueled expression, cityscape in the background, high-speed motion, cape flowing in the wind, intense focus, muzzle flashes, acrobatic and dynamic pose.",
    "Deadpool confronting a massive robot in the center of a city, clashing with the robot's powerful arms, buildings and cars in chaos, detailed robot machinery, explosive sparks flying, intense and gritty battle scene, Deadpool showing determination and agility.",
    "Deadpool dodging a hail of bullets in a dimly lit warehouse, weaving between crates and industrial equipment, sparks flying as bullets hit metal, focused and agile movement, high tension and suspense, shadows and dramatic lighting.",
    "Deadpool riding a motorcycle at high speed through a desert landscape, sand kicking up behind, intense focus, a fleet of enemy vehicles chasing, dynamic and thrilling chase scene, dramatic sky, motion and speed, dust clouds swirling.",
    "Deadpool defusing a complex bomb amidst a bustling marketplace, sweat on his brow, focused determination, intricate bomb details, anxious onlookers, vibrant marketplace atmosphere, high stakes and tension, sharp and detailed scene.",
    "Deadpool storming a bank, rescuing hostages, intense firefight, dramatic lighting, detailed bank interior, emotions running high, hostages expressing relief, chaotic and suspenseful scene, Deadpool exhibiting bravery and skill.",
    "Deadpool engaged in a sword fight on top of a moving train, sparks flying from clashing swords, intense expressions, detailed train and surroundings, high-speed action, blurring background, precise and controlled movements, wind whipping through the scene.",
    "Deadpool stopping a runaway bus in a crowded city street, using superhuman strength, bus skidding to a stop, dramatic motion blur, expressions of shock and awe from bystanders, bustling city backdrop, dynamic pose, high stakes and heroics."
]
# Define prompts and settings
negative_prompt = "blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs"
guidance_scale = 8.0
num_inference_steps = 30
RANDOMIZE_SEED = False  # Set to True to randomize the seed
fixed_seed = 3450349066
seed = fixed_seed if not RANDOMIZE_SEED else random.randint(0, 2**32 - 1)
torch.manual_seed(seed)  # Set the global seed for reproducibility
SPECIFIC_SCHEDULER = "DPM++ 2M Karras"

# Load the pre-trained SDXL model from HuggingFace
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define schedulers
schedulers = {
    "LMS": LMSDiscreteScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "Heun": HeunDiscreteScheduler,
    "DEIS": DEISMultistepScheduler,
    "DPM Solver++": DPMSolverSinglestepScheduler,
    "KDPM2": KDPM2DiscreteScheduler,
    "KDPM2 Ancestral": KDPM2AncestralDiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
    "PNDM": PNDMScheduler
}

# Ensure the output directory exists
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

def generate_images():
    selected_schedulers = [(SPECIFIC_SCHEDULER, schedulers[SPECIFIC_SCHEDULER])] if SPECIFIC_SCHEDULER else schedulers.items()
    
    for i, scene in enumerate(action_scenes, start=1):
        prompt = f"Deadpool, a superhero, {scene} Highly detailed, sharp, photorealism, cinematic lighting"
        for scheduler_name, scheduler_class in selected_schedulers:
            # Use the selected scheduler
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
            
            # Fix for the warning message
            if hasattr(pipe.scheduler.config, 'lower_order_final'):
                pipe.scheduler.config.lower_order_final = True
            
            # Generate the image
            start_time = time.time()
            generator_seed = seed + i  # Adjust the seed slightly for each generation
            if RANDOMIZE_SEED:
                generator_seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device="cuda").manual_seed(generator_seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            image = result.images[0]
            time_taken = time.time() - start_time
            
            # Save the image
            timestamp = get_timestamp()
            output_image_name = f"scene_{i}_{scheduler_name.replace(' ', '_')}_{timestamp}.png"
            output_image_path = os.path.join(output_dir, output_image_name)
            image.save(output_image_path)
            print(f"Generated image for scene {i} with {scheduler_name} saved as {output_image_path} (time taken: {time_taken:.2f}s)")

# Run the function to generate images
generate_images()

print(f"Used Seed: {seed if not RANDOMIZE_SEED else 'Randomized'}")