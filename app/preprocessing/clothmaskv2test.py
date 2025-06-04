from PIL import Image
import numpy as np
import cv2
from rembg import remove
import os

def process_cloth_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # Check if it's a file and has a valid image extension
        if os.path.isfile(input_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            output_mask_path = os.path.join(output_folder, file_name)
            
            try:
                # Open the image and process it
                img = Image.open(input_path).convert("RGBA")
                result = remove(img)

                # Create the mask
                alpha_channel = np.array(result)[:, :, 3]
                mask = np.zeros((alpha_channel.shape[0], alpha_channel.shape[1], 3), dtype=np.uint8)
                mask[alpha_channel > 0] = [255, 255, 255]  # White for non-transparent areas

                # Save the mask
                mask_img = Image.fromarray(mask)
                mask_img.save(output_mask_path)
                print(f"Processed: {file_name} -> Mask saved at {output_mask_path}")
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Define input and output folders
base_dir = os.getcwd()
data_path = os.path.join(base_dir, "uploads", "test", "cloth")
output_path = os.path.join(base_dir, "uploads", "test", "cloth-mask")

process_cloth_images(data_path, output_path)