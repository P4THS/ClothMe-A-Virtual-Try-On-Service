from PIL import Image, ImageFilter
import numpy as np
import os
from glob import glob

def generate_smoothed_mask(input_image_path, output_mask_path, target_values=(5, 6, 7), blur_radius=3):
    """Generate a smoothed mask for a single image."""
    try:
        # Load the image
        img = Image.open(input_image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)

        # Create a blank mask with the same dimensions as the input image
        mask = np.zeros_like(img_array, dtype=np.uint8)

        # Set mask to white (255) where pixel values match the target values
        for value in target_values:
            mask[img_array == value] = 255

        # Convert mask array to an image
        mask_image = Image.fromarray(mask)

        # Apply Gaussian blur to smooth edges
        smoothed_mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Generate final mask from smoothed image, considering all non-black pixels as white
        smoothed_array = np.array(smoothed_mask_image)
        final_mask = np.where(smoothed_array > 0, 255, 0).astype(np.uint8)  # Binary mask

        # Convert the final mask to an image
        final_mask_image = Image.fromarray(final_mask)
        final_mask_image.save(output_mask_path, format="JPEG")
        print(f"Final mask saved at: {output_mask_path}")

    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")

def process_all_images(input_folder, output_folder, target_values=(5, 6, 7), blur_radius=3):
    """Process all images in the input folder and save masks to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob(os.path.join(input_folder, fmt)))

    if not image_paths:
        print(f"No images found in the input folder: {input_folder}")
        return

    for input_image_path in image_paths:
        # Define output mask path
        image_name = os.path.basename(input_image_path).split('.')[0] + ".jpg"
        output_mask_path = os.path.join(output_folder, image_name)

        # Process the image
        generate_smoothed_mask(input_image_path, output_mask_path, target_values, blur_radius)

    print("Processing complete.")

# Example usage
if __name__ == "__main__":
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "uploads", "image-parse-v3")
    output_dir = os.path.join(base_dir, "uploads", "gt_cloth_warped_mask")
    process_all_images(data_path, output_dir)
