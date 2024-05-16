import os
import numpy as np
from PIL import Image
from skimage import filters, morphology

# Set the paths for image and mask saving
image_dir = "./Train/Image"
mask_dir = "./Train/Layer_Masks"

# Ensure the saving paths exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Function for edge detection to generate masks
def edge_detection(image):
    # Convert the image to grayscale
    gray_image = image.convert("L")
    # Perform edge detection using the Sobel operator
    edges = filters.sobel(np.array(gray_image))
    # Perform morphology operation to enhance the edges of cell nuclei
    edges = morphology.closing(edges, morphology.disk(3))
    # Use Otsu's method to compute the threshold
    threshold = filters.threshold_otsu(edges)
    # Generate mask using threshold segmentation
    mask = np.where(edges > threshold, 255, 0)
    return mask

# Traverse the original image folder and generate corresponding mask images
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)

    # Generate mask image
    mask = edge_detection(image)

    # Invert the mask image
    mask = 255 - mask

    # Convert to PIL image
    mask_pil = Image.fromarray(mask.astype(np.uint8))

    # Save the mask image
    mask_filename = os.path.splitext(filename)[0] + "_mask.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    mask_pil.save(mask_path)

print("Mask images have been generated and saved to the corresponding paths.")
