import numpy as np
from PIL import Image
import os

def resize_grayscale_repeat(image, new_width, new_height):
    """
    Resize a grayscale image using nearest neighbor replication to fit new dimensions.
    """
    old_height, old_width = image.shape

    # Calculate scaling factors
    scale_x = new_width // old_width
    scale_y = new_height // old_height

    # Expand the image by repeating pixels
    expanded_img = np.repeat(image, scale_y, axis=0)
    expanded_img = np.repeat(expanded_img, scale_x, axis=1)

    # Trim any excess pixels
    return expanded_img[:new_height, :new_width]

def resize_images_in_folder(folder_path, new_width, new_height):
    """
    Resize all images in the folder to the specified dimensions,
    replacing the original images.
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the file is an image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open the image and convert it to grayscale
                img = Image.open(file_path).convert('L')
                img_array = np.array(img)

                # Resize the image
                resized_array = resize_grayscale_repeat(img_array, new_width, new_height)

                # Save the resized image back to the same file
                resized_img = Image.fromarray(resized_array)
                resized_img.save(file_path)
                print(f"Resized and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Folder path and target dimensions
#folder_path = r"D:\assignments\fyp\final evaluation\website\uploads\test\image-parse-v3"  # Replace with your folder path
base_dir = os.getcwd()
folder_path = os.path.join(base_dir, "uploads", "test", "image-parse-v3")
new_width = 768
new_height = 1024

# Resize all images in the folder
resize_images_in_folder(folder_path, new_width, new_height)
