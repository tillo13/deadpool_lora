# Deadpool Lora Creator

Welcome to the **Deadpool Lora Creator** project. This initiative is geared towards generating high-quality, action-packed images of Deadpool using innovative AI technologies such as Stable Diffusion. The project stands out not just for its technical prowess but also for the creative prompt engineering and scheduling employed to capture the dynamic essence of Deadpool.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Python Scripts](#python-scripts)
  - [deadpool_v3.py](#deadpool_v3py)
  - [gather_pythons.py](#gather_pythonspy)
  - [mosaic.py](#mosaicpy)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Images](#generating-images)
  - [Gathering Python Files](#gathering-python-files)
  - [Creating Mosaics](#creating-mosaics)
- [Acknowledgments](#acknowledgments)

## Directory Structure

The directory structure of the project is designed for clear organization and ease of use:

```
.
├── deadpool_v3.py
├── gather_pythons.py
├── mosaic.py
└── [other folders and files]
```

## Python Scripts

### deadpool_v3.py

The script `deadpool_v3.py` is the cornerstone of this project. It is crafted to generate vivid and detailed action scenes featuring Deadpool. Each scene is meticulously described to capture the high-energy and dynamic nature of the character.

#### Key Features:
- **Action Scene Prompts**: Contains 10 well-crafted action scenes prompts featuring Deadpool, capturing various dynamic and intense scenarios.
- **Scheduler Options**: Leverages multiple sophisticated schedulers, offering nuanced control over the image generation process.
- **Model Utilization**: Utilizes the pre-trained Stable Diffusion XL model for producing high-detail, photorealistic images.
- **Customizability**: Provides options to adjust prompts, guidance scale, inference steps, and seeds to tailor image generation to specific requirements.

```python
import os
import time
import torch
import random
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionXLPipeline
# And more...
```

### gather_pythons.py

The script `gather_pythons.py` functions as a comprehensive collector of all Python files within the project's directory, excluding specified folders. This ensures better project organization and documentation.

#### Key Features:
- **Comprehensive Scanning**: Searches the entire project directory for Python files, ensuring no script is left untracked.
- **Directory Exclusion**: Intelligent exclusion of specific directories to keep the focus on relevant files.
- **Detailed Reporting**: Generates a timestamped file listing gathered Python files, enhancing project transparency and manageability.

### mosaic.py

The script `mosaic.py` is designed to create mosaic images, grouping generated images by their scheduler type. This not only aids in visualization but also serves as a useful tool for comparing the results produced by different schedulers.

#### Key Features:
- **Image Grouping**: Efficiently groups images by scheduler types, ensuring coherent mosaics.
- **Mosaic Creation**: Creates and titles mosaic images, saving them in a specified directory for easy review.
- **Enhanced Visualization**: Facilitates side-by-side comparison of different scenes and scheduler results.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.7+
- `torch`
- `Pillow`
- `diffusers`
- `os`
- `time`
- `random`
- `datetime`

## Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourrepo/deadpool-lora-creator.git
    cd deadpool-lora-creator
    ```

2. **Install the Required Packages**:
    ```sh
    pip install torch Pillow diffusers
    ```

## Usage

### Generating Images

To generate images, update the `deadpool_v3.py` script with your desired settings and prompts, then run the script:
    ```sh
    python deadpool_v3.py
    ```

### Gathering Python Files

Run the `gather_pythons.py` script to generate a detailed list of all Python files in the project directory:
    ```sh
    python gather_pythons.py
    ```

### Creating Mosaics

After generating images, create mosaic representations using the `mosaic.py` script:
    ```sh
    python mosaic.py
    ```

## Acknowledgments

This project leverages several powerful tools and libraries:
- **Diffusers**: For providing the versatile `StableDiffusionXLPipeline`.
- **HuggingFace**: For hosting the Stable Diffusion XL model.
- **Pillow**: For its robust image processing capabilities.
- **The Open-Source Community**: For contributions and support in advancing AI and machine learning technologies.

Feel free to contribute, report issues, or fork the repository to add new features or improve existing ones. Your participation is highly appreciated!