from PIL import Image
import os
import shutil

def resize_images(input_folder, output_folder, width=192, height=256):
    try:
        # Ensure the output folder exists and clear it
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)  # Remove all files and subdirectories
        os.makedirs(output_folder, exist_ok=True)  # Recreate the output folder

        # Process all image files in the input folder
        for file_name in os.listdir(input_folder):
            input_path = os.path.join(input_folder, file_name)
            
            # Check if it's a valid image file
            if os.path.isfile(input_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                output_path = os.path.join(output_folder, file_name)
                
                # Open, resize, and save the image
                with Image.open(input_path) as img:
                    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                    resized_img.save(output_path)
                    print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error: {e}")

# Define input and output folders
base_dir = os.getcwd()
input_folder = os.path.join(base_dir, "uploads", "test", "image")
output_folder = os.path.join(base_dir, "preprocessing",  "CIHP_PGNcode", "datasets")

# Resize images
resize_images(input_folder, output_folder)
