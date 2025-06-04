import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)

    # head & lower-body parse masks (if you still want to restore those regions)
    parse_head  = ((parse_array == 4)  | (parse_array == 13)).astype(np.float32)
    parse_lower = ((parse_array == 9)  | (parse_array == 12)
                 | (parse_array == 16) | (parse_array == 17)
                 | (parse_array == 18) | (parse_array == 19)).astype(np.float32)

    # start from the original image
    agnostic      = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # normalize limb lengths
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    center   = (pose_data[9] + pose_data[12]) / 2
    pose_data[9]  = center + (pose_data[9]  - center) / length_b * length_a
    pose_data[12] = center + (pose_data[12] - center) / length_b * length_a
    r = int(length_a / 16) + 1

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        x, y = pose_data[i]
        agnostic_draw.ellipse((x-r*5, y-r*5, x+r*5, y+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if ((pose_data[i-1,0]==0 and pose_data[i-1,1]==0)
        or   (pose_data[i,0]==0 and pose_data[i,1]==0)):
            continue
        agnostic_draw.line([tuple(pose_data[i-1]), tuple(pose_data[i])],
                            'gray', width=r*10)
        x, y = pose_data[i]
        agnostic_draw.ellipse((x-r*5, y-r*5, x+r*5, y+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        x, y = pose_data[i]
        agnostic_draw.ellipse((x-r*3, y-r*6, x+r*3, y+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]],  'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5,12]],  'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9,12]],  'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2,5,12,9]],
                           'gray', 'gray')

    # mask neck
    x, y = pose_data[1]
    agnostic_draw.rectangle((x-r*7, y-r*7, x+r*7, y+r*7), 'gray', 'gray')

    # restore head & lower-body to original pixels
    agnostic.paste(img, None,
                   Image.fromarray((parse_head * 255).astype('uint8'), 'L'))
    agnostic.paste(img, None,
                   Image.fromarray((parse_lower * 255).astype('uint8'), 'L'))

    # --- NEW: cover *all* parsed cloth pixels with gray ---
    cloth_mask = ((parse_array == 5)  # upper-clothes
                | (parse_array == 6)  # dress
                | (parse_array == 7)).astype(np.uint8)
    gray_fill = Image.new('RGB', agnostic.size, 'gray')
    cloth_l   = Image.fromarray(cloth_mask * 255, mode='L')
    agnostic.paste(gray_fill, mask=cloth_l)

    return agnostic

if __name__ == "__main__":
    base_dir    = os.getcwd()
    data_path   = osp.join(base_dir, "uploads", "test")
    output_path = osp.join(data_path, "agnostic-v3.2")
    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        if not im_name.lower().endswith('.jpg'):
            continue

        # load pose keypoints
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
            kp = np.array(pose_label['people'][0]['pose_keypoints_2d'])
            pose_data = kp.reshape(-1, 3)[:, :2]
        except (IndexError, FileNotFoundError):
            print(f"Skipping {pose_name}")
            continue

        # load images
        im       = Image.open(osp.join(data_path, 'image', im_name)).convert('RGB')
        label_im = Image.open(osp.join(data_path, 'image-parse-v3',
                                       im_name.replace('.jpg', '.png')))

        agnostic = get_img_agnostic(im, label_im, pose_data)
        agnostic.save(osp.join(output_path, im_name))
