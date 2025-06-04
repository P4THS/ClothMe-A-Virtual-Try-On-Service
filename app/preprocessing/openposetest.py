# import os
# import subprocess

# def generate_skeleton(image_path, output_dir_img, output_dir_json, openpose_dir):
#     image_path = os.path.abspath(image_path)
#     output_dir_img = os.path.abspath(output_dir_img)
#     output_dir_json = os.path.abspath(output_dir_json)
#     openpose_dir = os.path.abspath(openpose_dir)

#     command = f'cd /d "{openpose_dir}" && bin\\OpenPoseDemo.exe --image_dir "{os.path.dirname(image_path)}" --hand --write_images "{output_dir_img}" --write_json "{output_dir_json}" --display 0 --disable_blending'

#     print("Running command:", command)

#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()

#     if process.returncode != 0:
#         print(f"Error: {stderr.decode('utf-8')}")
#     else:
#         print(f"Skeleton generated and saved in '{output_dir_img}'")

# image_path = r"D:\assignments\fyp\final evaluation\website\uploads\test\image\random.jpg" 
# output_dir_img = r"D:\assignments\fyp\final evaluation\website\uploads\test\openpose_img"                
# output_dir_json = r"D:\assignments\fyp\final evaluation\website\uploads\test\openpose_json"                
# openpose_dir = "D:/downloads/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose" 

# generate_skeleton(image_path, output_dir_img, output_dir_json, openpose_dir)

import os
import subprocess

def generate_skeleton(image_path, output_dir_img, output_dir_json, openpose_dir):
    image_path = os.path.abspath(image_path)
    output_dir_img = os.path.abspath(output_dir_img)
    output_dir_json = os.path.abspath(output_dir_json)
    openpose_dir = os.path.abspath(openpose_dir)

    command = (
        f'cd "{openpose_dir}" && conda run -n test && ./build/examples/openpose/openpose.bin '
        f'--image_dir "{os.path.dirname(image_path)}" --hand '
        f'--write_images "{output_dir_img}" --write_json "{output_dir_json}" '
        f'--display 0 --disable_blending'
    )

    print("Running command:", command)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash")
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        print(f"Skeleton generated and saved in '{output_dir_img}'")

# Set base directory
base_dir = os.getcwd()
image_path = os.path.join(base_dir, "uploads", "test", "image", "random.jpg")
output_dir_img = os.path.join(base_dir, "uploads", "test", "openpose_img")
output_dir_json = os.path.join(base_dir, "uploads", "test", "openpose_json")
openpose_dir = os.path.join("preprocessing", "openpose")


generate_skeleton(image_path, output_dir_img, output_dir_json, openpose_dir)
