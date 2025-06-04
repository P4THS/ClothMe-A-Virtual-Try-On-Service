from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import os
import json
from os import path as osp
# Optional: uncomment if you want to do morphological closing to clean the cloth mask
# import cv2

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)

    # 1) Original head & lower‐body parsing masks (if you still need to blank them)
    parse_head = ((parse_array == 4) | (parse_array == 13)).astype(np.float32)
    parse_lower = ((parse_array == 9)  | (parse_array == 12)
                 | (parse_array == 16) | (parse_array == 17)
                 | (parse_array == 18) | (parse_array == 19)).astype(np.float32)

    # 2) Blank RGB canvas
    agnostic = Image.new('RGB', (768, 1024), 'black')
    black    = Image.new('RGB', (768, 1024), 'black')
    draw     = ImageDraw.Draw(agnostic)

    # 3) Normalize limb lengths for consistent stroke widths
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])   # upper arm
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])  # upper leg
    center   = (pose_data[9] + pose_data[12]) / 2
    # scale hip points so leg segment length == arm segment length
    pose_data[9]  = center + (pose_data[9]  - center) / length_b * length_a
    pose_data[12] = center + (pose_data[12] - center) / length_b * length_a
    r = int(length_a / 16) + 1

    # 4) Draw pose‐based white shapes (arms, torso, neck)
    # Arms: upper arm line + joints
    draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)
    for i in [2, 5]:
        x, y = pose_data[i]
        draw.ellipse((x-r*5, y-r*5, x+r*5, y+r*5), 'white', 'white')
    # Forearms/wrists
    for i in [3, 4, 6, 7]:
        if (pose_data[i-1,0]==0 and pose_data[i-1,1]==0) or (pose_data[i,0]==0 and pose_data[i,1]==0):
            continue
        draw.line([tuple(pose_data[i-1]), tuple(pose_data[i])], 'white', width=r*10)
        x, y = pose_data[i]
        draw.ellipse((x-r*5, y-r*5, x+r*5, y+r*5), 'white', 'white')

    # Torso polygon + hip joints
    for i in [9, 12]:
        x, y = pose_data[i]
        draw.ellipse((x-r*3, y-r*6, x+r*3, y+r*6), 'white', 'white')
    draw.line([tuple(pose_data[2]),  tuple(pose_data[9])],  'white', width=r*6)
    draw.line([tuple(pose_data[5]),  tuple(pose_data[12])], 'white', width=r*6)
    draw.line([tuple(pose_data[9]),  tuple(pose_data[12])], 'white', width=r*12)
    draw.polygon([tuple(pose_data[i]) for i in [2,5,12,9]], 'white', 'white')

    # Neck
    x, y = pose_data[1]
    draw.rectangle((x-r*7, y-r*7, x+r*7, y+r*7), 'white', 'white')

    # 5) Optionally blank out head & lower-body via parse
    agnostic.paste(black, None, Image.fromarray((parse_head*255).astype('uint8'), 'L'))
    agnostic.paste(black, None, Image.fromarray((parse_lower*255).astype('uint8'), 'L'))

    # 6) Build cloth mask from semantic parse (upper-clothes, dress, coat)
    cloth_mask = ((parse_array == 5)  # upper-clothes
                | (parse_array == 6)  # dress
                | (parse_array == 7)  # coat
               ).astype(np.uint8)

    # Optional: clean up small holes in cloth_mask
    # kernel = np.ones((15,15), np.uint8)
    # cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)

    # 7) Paste solid white over every cloth pixel
    white_fill = Image.new('RGB', agnostic.size, 'white')
    cloth_l    = Image.fromarray(cloth_mask * 255, mode='L')
    agnostic.paste(white_fill, mask=cloth_l)

    return agnostic


if __name__ == "__main__":
    base_dir    = os.getcwd()
    data_path   = osp.join(base_dir, "uploads", "test")
    output_path = osp.join(data_path, "agnostic-mask")
    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        if not im_name.lower().endswith('.jpg'):
            continue

        # Load pose keypoints
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
            pose_kps = np.array(pose_label['people'][0]['pose_keypoints_2d'])
            pose_data = pose_kps.reshape(-1, 3)[:, :2]
        except (IndexError, FileNotFoundError):
            print(f"Skipping {pose_name}: no valid pose")
            continue

        # Load images
        im        = Image.open(osp.join(data_path, 'image', im_name)).convert('RGB')
        label_im  = Image.open(osp.join(data_path, 'image-parse-v3', im_name.replace('.jpg','.png')))

        # Generate & save
        agn = get_img_agnostic(im, label_im, pose_data)
        agn.save(osp.join(output_path, im_name))
