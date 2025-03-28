import argparse

import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def compress_image(input_path, output_path, quality=85):
    """Compresses an image using Pillow."""

    with Image.open(input_path) as image:
        image.save(output_path, optimize=True, quality=quality)


def load_image(image_path):
  image = Image.open(image_path)
  transform = transforms.Compose([
      transforms.ToTensor()
  ])
  return transform(image).unsqueeze(0)

# Function to calculate PSNR
def calculate_psnr(img1, img2):
  # Ensure both images are on the same device
  img1 = img1.to(img2.device)

  mse = torch.mean((img1 - img2) ** 2)
  if mse == 0:
      return float('inf')
  psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
  return psnr.item()

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        self.enc5 = self._block(512, 1024)
        
        # Bottleneck
        self.bottleneck = self._block(1024, 2048)
        
        # Decoder
        self.dec5 = self._block(2048 + 1024, 1024)
        self.dec4 = self._block(1024 + 512, 512)
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc5))
        
        # Decoder path
        dec5 = self._up_concat(bottleneck, enc5, self.dec5)
        dec4 = self._up_concat(dec5, enc4, self.dec4)
        dec3 = self._up_concat(dec4, enc3, self.dec3)
        dec2 = self._up_concat(dec3, enc2, self.dec2)
        dec1 = self._up_concat(dec2, enc1, self.dec1)
        
        # Final layer
        return self.final(dec1)
    
    def _up_concat(self, x, skip, decoder):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return decoder(torch.cat([x, skip], dim=1))

def deep_image_prior(image_path, num_iterations=10000):
  # Load the corrupted image
  corrupted_image = load_image(image_path)

  original_image = load_image('og_small.jpg')

  # Initialize the network and optimizer
  net = UNet(3,3)
  optimizer = optim.Adam(net.parameters(), lr=0.000001)

  # Create a random noise input
  input_noise = torch.randn_like(corrupted_image)
  # Define the loss function - MSE
  criterion = nn.MSELoss()

  # Store Loss and PSNR values
  loss_values = []
  psnr_values = []

  psnr_max = 15

  # Training loop
  for i in range(num_iterations + 1):
    print(i,end='\r')
    optimizer.zero_grad()

    # Forward pass through the network
    output_image = net(input_noise)

    # Compute the loss between output and corrupted image
    loss = criterion(output_image, corrupted_image)

    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()

    # Print and Save Image
    # if i % 500 == 0:
    loss_values.append(loss.item())
    psnr_value = calculate_psnr(output_image * 255.0, corrupted_image * 255.0)
    psnr_values.append(psnr_value)
    if i % 1000 == 0:
      print(f"Iteration {i}/{num_iterations}, Loss: {loss.item():.10f}, PSNR: {psnr_value:.2f} dB")
      save_image(output_image.cpu().detach(), f"restored_og{i}.png")

    pnsr_original = calculate_psnr(output_image * 255.0, original_image * 255.0)

    if pnsr_original > psnr_max:
      psnr_max = pnsr_original
      save_image(output_image.cpu().detach(), f"restored_max.png")
      print("The parameters at interation: "+str(i)+" with psnr: "+str(psnr_max), end ='\r')


  # Plotting the results
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(loss_values, label='Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Loss over Iterations')

  plt.subplot(1, 2, 2)
  plt.plot(psnr_values, label='PSNR', color='orange')
  plt.xlabel('Iteration')
  plt.ylabel('PSNR (dB)')
  plt.title('PSNR over Iterations')

  plt.tight_layout()
  plt.show()

# Call main function


def main():
    parser = argparse.ArgumentParser(description="Run Deep Image Prior with a specified image path.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of iterations for optimization")

    args = parser.parse_args()

    deep_image_prior(args.image_path, args.num_iterations)

if __name__ == "__main__":
    main()
