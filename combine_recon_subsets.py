import numpy as np
import matplotlib.pyplot as plt
import glob

# Path pattern to your subset reconstructions
recon_path_pattern = ""                     #replace with your output path from previous script

# List all subset files
subset_files = sorted(glob.glob(recon_path_pattern))

# Load and accumulate images
accum_image = None
num_subsets = len(subset_files)

for file in subset_files:
    data = np.load(file)
    img = data["image"]
    if accum_image is None:
        accum_image = np.zeros_like(img)
    accum_image += img

# Average over subsets
final_image = accum_image / num_subsets

# Reshape to 2D if necessary
final_image_2d = final_image.reshape((500, 500))  # Adjust shape to your image size

# Display combined image
plt.figure(figsize=(8, 8))
plt.imshow(final_image_2d, cmap="gray", origin="lower")
plt.colorbar(label="Intensity")
plt.title("Combined Final Reconstructed Image")
plt.show()
