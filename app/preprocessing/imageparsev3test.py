# import os
# import subprocess
# import sys
# from glob import glob
# import shutil

# # Define constants
# CONDA_ENV_NAME = "imageparsev2"
# SCRIPT_PATH = r"D:\assignments\fyp\final evaluation\CIHP_PGNcode\inference_pgntest.py"
# INPUT_FOLDER = r"D:\assignments\fyp\final evaluation\CIHP_PGNcode\datasets"
# TEMP_FOLDER = r"D:\assignments\fyp\final evaluation\CIHP_PGNcode\datasets\temp"

# def get_image_paths(folder):
#     """Get all image files in the folder."""
#     supported_formats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
#     image_paths = []
#     for fmt in supported_formats:
#         image_paths.extend(glob(os.path.join(folder, fmt)))
#     return image_paths

# def process_images(image_paths):
#     """Process each image by placing it in a temp folder and running the script."""
#     if not os.path.exists(TEMP_FOLDER):
#         os.makedirs(TEMP_FOLDER)

#     for image_path in image_paths:
#         print(f"Processing {image_path}...")
#         temp_image_path = os.path.join(TEMP_FOLDER, os.path.basename(image_path))
        
#         # Copy the image to the temp folder
#         shutil.copy(image_path, temp_image_path)
        
#         # Retry mechanism
#         while True:
#             command = f'conda run -n {CONDA_ENV_NAME} python "{SCRIPT_PATH}" --image-path "{TEMP_FOLDER}"'
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = process.communicate()

#             if process.returncode == 0:
#                 print(f"Successfully processed {image_path}")
#                 break  # Exit the loop on success
#             else:
#                 print(f"Error processing {image_path}: {stderr.decode('utf-8')}")
#                 print("Retrying...")
        
#         # Clean up: Delete the image from the temp folder
#         if os.path.exists(temp_image_path):
#             os.remove(temp_image_path)

# def main():
#     image_paths = get_image_paths(INPUT_FOLDER)
#     if not image_paths:
#         print(f"No images found in the folder: {INPUT_FOLDER}")
#         sys.exit(1)

#     print(f"Found {len(image_paths)} images. Starting processing...")
#     process_images(image_paths)
#     print("Processing complete!")

#     # Clean up: Remove the temp folder if empty
#     if os.path.exists(TEMP_FOLDER) and not os.listdir(TEMP_FOLDER):
#         os.rmdir(TEMP_FOLDER)

# if __name__ == "__main__":
#     main()

import os
import subprocess
import sys
from glob import glob
import shutil

# Define constants
CONDA_ENV_NAME = "cihp"

# Get the base directory dynamically
base_dir = os.getcwd()
SCRIPT_PATH = os.path.join("preprocessing", "CIHP_PGNcode", "inference_pgntest.py")
INPUT_FOLDER = os.path.join("preprocessing", "CIHP_PGNcode", "datasets")
TEMP_FOLDER = os.path.join("preprocessing", "CIHP_PGNcode", "datasets", "temp")


def get_image_paths(folder):
    """Get all image files in the folder."""
    supported_formats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob(os.path.join(folder, fmt)))
    return image_paths


def process_images(image_paths):
    """Process each image by placing it in a temp folder and running the script."""
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    for image_path in image_paths:
        print(f"Processing {image_path}...")
        temp_image_path = os.path.join(TEMP_FOLDER, os.path.basename(image_path))

        # Copy the image to the temp folder
        shutil.copy(image_path, temp_image_path)
        print(SCRIPT_PATH)

        # Retry mechanism
        while True:
            command = f'conda run -n {CONDA_ENV_NAME} python3 "{SCRIPT_PATH}" --image-path "{TEMP_FOLDER}"'

            print(f"Executing: {command}")

            # Run the command and stream output in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=sys.stdout,  # Direct output to console
                stderr=sys.stderr   # Direct errors to console
            )

            process.wait()  # Wait for process to complete

            if process.returncode == 0:
                print(f"Successfully processed {image_path}")
                break  # Exit the loop on success
            else:
                print(f"Error processing {image_path}, retrying...")

        # Clean up: Delete the image from the temp folder
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def main():
    image_paths = get_image_paths(INPUT_FOLDER)
    if not image_paths:
        print(f"No images found in the folder: {INPUT_FOLDER}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Starting processing...")
    process_images(image_paths)
    print("Processing complete!")

    # Clean up: Remove the temp folder if empty
    if os.path.exists(TEMP_FOLDER) and not os.listdir(TEMP_FOLDER):
        os.rmdir(TEMP_FOLDER)


if __name__ == "__main__":
    main()
