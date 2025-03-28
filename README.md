# UNet Skip Connection Implementation

## Overview

This project implements a UNet architecture with skip connections for image restoration following lossy compression. It is based on the Deep Image Prior methodology, where an untrained convolutional network is optimized to restore degraded images without requiring pre-trained datasets.

## Project Structure

- **Unet\_skip\_connect\_Implementation.py** - Python script extracted from the Jupyter Notebook for direct execution.

## Methodology

- A CNN with a UNet-like architecture and skip connections is used for image restoration.
- The model is trained on a single degraded image, leveraging its architectural properties for restoration.
- Performance is evaluated based on PSNR (Peak Signal-to-Noise Ratio) metrics.

## Installation & Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/Prathamesh-Pawar/image-dip.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```sh
   python Unet_skip_connect_Implementation.py --image_path path/to/your/image.jpg --num_iterations 5000

   ```

## Results

- The model effectively restores compressed images to a high-quality approximation of the original.
- Best results are observed after approximately 35,000 iterations.

## References

- Deep Image Prior: Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2020). International Journal of Computer Vision.
- Explanation of Skip Connections: The AI Summer.
- Image Sources: Freepik.com

