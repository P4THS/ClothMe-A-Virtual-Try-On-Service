import subprocess
import os
import glob
import re

# Define the path to your working directory and input/output directories
# working_dir = r"D:\assignments\fyp\code\detectron2-main\detectron2-main\projects\DensePose"
# input_dir = r"D:/assignments/fyp/final evaluation/website/uploads/image/"
# output_dir = r"D:/assignments/fyp/final evaluation/website/uploads/image-densepose/"

base_dir = os.getcwd()
input_dir = os.path.join(base_dir, "uploads", "image")
output_dir = os.path.join(base_dir, "uploads", "image-densepose")
working_dir = os.path.join("preprocessing", "detectron2", "projects", "DensePose")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the base command
base_command = [
    "conda", "run", "-n", "test", "python", "-u", "apply_net.py", "show",
    "configs/densepose_rcnn_R_50_FPN_WC1M_s1x.yaml",  # Config file
    "model_final_48a9d9.pkl",  # Model file
    
]

# Change the current working directory to your project directory
os.chdir(working_dir)

# Get a list of all image files in the input directory
image_files = glob.glob(os.path.join(input_dir, "*.jpg"))

# Process each image
for image_file in image_files:
    # Define the output file path
    file_name = os.path.basename(image_file)
    output_file = os.path.join(output_dir, file_name)

    # Construct the command for the current image
    command = base_command + [
        image_file,  # Input image
        "dp_segm",  # Task
        "-v",  # Verbose flag
        "--output", output_file  # Output file
    ]
    print(command)
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print(f"Error processing {file_name}:\n{result.stderr}")
    else:
        print(f"Successfully processed {file_name}:\n{result.stdout}")

folder_path = output_dir

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Construct the full path of the file
    old_file_path = os.path.join(folder_path, file_name)

    # Check if it's a file
    if os.path.isfile(old_file_path):
        # Use regex to remove the `.xxxx` pattern from the file name
        new_file_name = re.sub(r'\.\d{4}', '', file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

print("All files have been renamed.")