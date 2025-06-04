from flask import Flask, send_file, render_template, request, flash, redirect, url_for, jsonify, Response, send_from_directory, abort
import requests
import os
import re
from werkzeug.utils import secure_filename
import subprocess
import shutil
from PIL import Image
import torch
import json
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

base_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(base_dir, "uploads")
PREPROCESSING_FOLDER = os.path.join(base_dir, "preprocessing")


for item in os.listdir(UPLOAD_FOLDER):
    item_path = os.path.join(UPLOAD_FOLDER, item)

    # Check if the item is a folder
    if os.path.isdir(item_path):
        # Remove the folder and its contents
        shutil.rmtree(item_path)

IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'image')
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'cloth-mask')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'agnostic-mask')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'agnostic-v3.2')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'gt_cloth_warped_mask')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'image-densepose')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'image-parse-agnostic-v3.2')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'openpose_img')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'openpose_json')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'image-parse-v3')
os.makedirs(CLOTH_FOLDER, exist_ok=True)
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'cloth')

TOTAL_EPOCHS = 1
TOTAL_FILES = 1

ALLOWED_EXTENSIONS = {'jpg'}

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLOTH_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CKPT_PATH = './StableVITON/ckpts/VITONHD.ckpt'
SAVE_DIR = './StableVITON/ckpts/'

@app.route('/request_weights', methods=['GET'])
def request_weights_file():
    # send checkpoint binary
    resp = send_file(
        CKPT_PATH,
        as_attachment=True,
        download_name=os.path.basename(CKPT_PATH),
        mimetype='application/octet-stream'
    )
    # attach metadata headers
    resp.headers['X-Total-Epochs'] = str(TOTAL_EPOCHS)
    resp.headers['X-Total-Files'] = str(TOTAL_FILES)
    return resp

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global CKPT_PATH, SAVE_DIR
    # receive a binary file upload and save it directly
    if 'file' not in request.files:
        abort(400, "Missing file in request")
    file = request.files['file']
    # ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    # save incoming file with original filename or timestamped
    filename = file.filename or f"updated_{int(time.time())}.ckpt"
    out_path = os.path.join(SAVE_DIR, filename)
    file.save(out_path)
    temp = CKPT_PATH
    CKPT_PATH = SAVE_DIR + filename
    os.remove(temp)
    return jsonify({'status': 'ok', 'saved_to': out_path})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_base_name(file_name):
    return re.sub(r'_(cloth|person)', '', file_name)

@app.route('/')
def landingpage():
    return render_template('landingpage.html')

@app.route('/options')
def options():
    return render_template('options.html')

@app.route('/trainingv2', methods=['GET', 'POST'])
def trainingv2():
    
    if request.method == 'POST':
        files = request.files.getlist('fileInput')
        print(files)
        valid_files = [f for f in files if allowed_file(f.filename)]
        
        # Check for valid file pairs
        cloth_files = {}
        person_files = {}
        
        for file in valid_files:
            base_name = get_base_name(secure_filename(file.filename))
            base_name = base_name[:-4]
            print(base_name)
            if '_cloth' in file.filename:
                cloth_files[base_name] = file
            elif '_person' in file.filename:
                person_files[base_name] = file

        missing_pairs = set(cloth_files.keys()) ^ set(person_files.keys())
        if missing_pairs:
            flash(f"Error: Unpaired images detected for names: {', '.join(missing_pairs)}", 'error')
            try:
                files = [
                    file for file in os.listdir(CLOTH_FOLDER)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                ]
            except FileNotFoundError:
                files = []  # If folder doesn't exist, set files to empty

            return render_template('trainingv2.html', files=files)
        
        for base_name in cloth_files:
            cloth_file = cloth_files[base_name]
            person_file = person_files[base_name]
            
            cloth_file.save(os.path.join(CLOTH_FOLDER, f"{base_name}.jpg"))
            person_file.save(os.path.join(IMAGE_FOLDER, f"{base_name}.jpg"))

        
        print("starting the preprocessing")
        
    
        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/openpose.py"'

        
    
        
    
   
        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        empty_files = []
        directory_path = "./uploads/openpose_json"
    
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):  # Ensure we only process JSON files
                file_path = os.path.join(directory_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        
                        if isinstance(data, dict) and data.get("people") == []:
                            empty_files.append(filename)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {filename}: {e}")

        hh = 0
        for filename in empty_files:
            filepath = os.path.join("./uploads/openpose_json", filename)
            os.remove(filepath)
            empty_files[hh] = filename[:-14] + 'rendered.png'
            hh+=1
        
        hh=0

        for filename in empty_files:
            filepath = os.path.join("./uploads/openpose_img", filename)
            os.remove(filepath)
            empty_files[hh] = filename[:-12] + 'person.jpg'
            hh+=1


        print("starting the preprocessing first done")

        if empty_files.__len__():
            flash(f"Error: No person detected in images detected for names: {', '.join(empty_files)}", 'error')
            try:
                files = [
                    file for file in os.listdir(CLOTH_FOLDER)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                ]
            except FileNotFoundError:
                files = []  # If folder doesn't exist, set files to empty

            return render_template('trainingv2.html', files=files)

        
        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/densepose.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/clothmaskv2.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageresizer.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageparsev3.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/zerorderhold.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageparseagnosticv3.2.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/agnosticv3.2.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/agnosticmask.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/gtwarpedclothmask.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        folder_path = CLOTH_FOLDER
        files = os.listdir(folder_path) 

        for file in files:
            cfolder = "uploads/cloth/"+file
            ifolder = "uploads/image/"+file
            mfolder = "uploads/gt_warped_cloth_mask/"+file
            command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/matching.py" ' + cfolder + ' ' + ifolder + ' ' + mfolder

            print("Running command:", command)

            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Stream stdout/stderr
            for line in process.stdout:
                print(line, end="")
            for line in process.stderr:
                print(line, end="")
            process.wait()

            if process.returncode != 0:
                flash(f"Error: Cloth does not match with person wearing it for file: {(file)}", 'error')
                try:
                    files = [
                        file for file in os.listdir(CLOTH_FOLDER)
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                    ]
                except FileNotFoundError:
                    files = []  # If folder doesn't exist, set files to empty

                return render_template('trainingv2.html', files=files)

        
        flash('Success: File uploaded successfully!', 'success')
        try:
            files = [
                file for file in os.listdir(CLOTH_FOLDER)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ]
        except FileNotFoundError:
            files = []  # If folder doesn't exist, set files to empty

        return render_template('trainingv2.html', files=files)
    
    try:
        files = [
            file for file in os.listdir(CLOTH_FOLDER)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
    except FileNotFoundError:
        files = []  # If folder doesn't exist, set files to empty

    print(files)

    return render_template('trainingv2.html', files=files)

@app.route('/try_on', methods=['GET', 'POST'])
def try_on():
    if request.method == 'POST':
        # Handle try-on logic here
        return "Virtual try-on processing..."
    clothes = os.listdir(CLOTH_FOLDER)  # List all files in clothes folder
    return render_template('try_on.html', clothes=clothes)
    

@app.route('/uploads/cloth/<filename>')
def uploaded_file(filename):
    return send_from_directory(CLOTH_FOLDER, filename)



@app.route('/sendtraining', methods=['POST'])
def send_training():
    # Get JSON data from the request
    data = request.get_json()
    UPLOAD_ROOT = "./uploaded_files/train"

   
    # Extract data
    selected_files = data.get('selectedFiles', [])
    epochs = data.get('epochs', 0)
    batch_size = data.get('batchSize', 0)
    learning_rate = data.get('learningRate', 0.0)

    # Print data for debugging
    print("Selected Files:", selected_files)
    print("Training Parameters:")
    print("  Epochs:", epochs)
    print("  Batch Size:", batch_size)
    print("  Learning Rate:", learning_rate)

    files_to_send = []
    folder_names = ['agnostic-mask', 'agnostic-v3.2', 'cloth', 'cloth-mask', 'gt_cloth_warped_mask', 'image', 'image-densepose', 'image-parse-agnostic-v3.2', 'image-parse-v3', 'openpose_img', 'openpose_json']
    for file_name in selected_files:
        file_name = file_name[:-4]
        file_tuple = []  # Collect matching files from all 11 folders
        for i in range(0, 11):  # Assuming folders are named 1 to 11
            folder_path = os.path.join('./uploads', folder_names[i])
            if i == 0:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 1:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 2:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 3:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 4:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 5:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 6:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 7:
                file_path = os.path.join(folder_path, file_name + '.png')
            if i == 8:
                file_path = os.path.join(folder_path, file_name + '.png')
            if i == 9:
                file_path = os.path.join(folder_path, file_name + '_rendered.png')
            if i == 10:
                file_path = os.path.join(folder_path, file_name + '_keypoints.json')

            if os.path.exists(file_path):
                file_tuple.append(('file', open(file_path, 'rb')))
        if len(file_tuple) == 11:  # Only send tuples of 11 files
            files_to_send.extend(file_tuple)

    print(files_to_send)

    # Send request to remote serve



    

        # Delete all existing folders and files in UPLOAD_ROOT
    for item in os.listdir(UPLOAD_ROOT):
        item_path = os.path.join(UPLOAD_ROOT, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # Validate parameters
    if not epochs or not batch_size or not learning_rate:
        return jsonify({"error": "Missing training parameters"}), 400

    # Save received files and maintain folder structure
    i = 0
    print(f"Received files: {request.files}")

    files = request.files.getlist('file')

    # List to collect filenames in the 'cloth' folder
    cloth_filenames = []

    # First, save all the files and collect cloth filenames
    for file_tuple in files_to_send:
        if file_tuple:
            folder_name = folder_names[i % 11]
            file_object = file_tuple[1]  # Extract the file object
            file_name = os.path.basename(file_object.name)  # Get the correct filename

            # Ensure folder exists
            folder_path = os.path.join(UPLOAD_ROOT, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Save file properly
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'wb') as f:
                f.write(file_object.read())  # Write file contents to disk

            print(f"Saved {file_name} to {folder_path}")

            # Collect filenames from 'cloth' folder
            if folder_name == 'cloth':
                cloth_filenames.append(file_name)

        i += 1

    # After all files are saved, write the filenames to a .txt file in the requested format
    if cloth_filenames:
        train_file_path = os.path.join('./uploaded_files', 'train_pairs.txt')
        test_file_path = os.path.join('./uploaded_files', 'test_pairs.txt')

        with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file:
            # Write filenames to train and test files
            for idx, filename in enumerate(cloth_filenames):
                train_file.write(f"{filename} {filename}\n")
                if idx < len(cloth_filenames) * 0.1:  # Write 10% to test file
                    test_file.write(f"{filename} {filename}\n")

        print(f"Train file created at: {train_file_path}")
        print(f"Test file created at: {test_file_path}")

    # Print for debugging
    print("Training Parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")

    total_epochs += epochs
    total_files += len(cloth_filenames)

    # Run the specified command using subprocess
    command = 'CUDA_VISIBLE_DEVICES=0 conda run -n StableVITON python ./StableVITON/train.py \
    --config_name VITONHD \
    --transform_size shiftscale3 hflip \
    --transform_color hsv bright_contrast \
    --save_name Base_test'


    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        print("Command executed successfully")

    return jsonify({
        "message": "Files and parameters received successfully, and training command executed.",
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
    })





@app.route('/uploads/test/output<filename>')
def uploaded_filev2(filename):
    return send_from_directory('./uploads/test/output', filename)

@app.route('/uploads/test/image<filename>')
def uploaded_filev3(filename):
    return send_from_directory('./uploads/test/image', filename)

@app.route('/generatetryon', methods=['POST'])
def receive_training_files():
    # Get training parameters
    upload = './uploaded_files/test'
    os.makedirs(upload, exist_ok=True)
    # Save received files and maintain folder structure
    folder_names = ['agnostic-mask', 'agnostic-v3.2', 'cloth', 'cloth-mask', 'gt_cloth_warped_mask', 'image', 'image-densepose', 'image-parse-agnostic-v3.2', 'image-parse-v3', 'openpose_img', 'openpose_json']
    i = 0
    print()
    print()
    print()
    print(request.files)
          
    files = request.files.getlist('file')

    for file in files:
        print(file)
        folder_name = folder_names[i % 11]
        file_name = file.filename

        # Ensure folder exists
        folder_path = os.path.join(upload, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save fil
        file_path = os.path.join(folder_path, file_name)
        file.save(file_path)
        i+=1
        # Maintain folder structure for response


    return jsonify({
        "message": "Files and parameters received successfully.",
        
    })

@app.route('/generate', methods=['POST'])
def generate_tryon():
    user_photo = request.files['userPhoto']
    selected_cloth = request.form['selectedCloth']

    folder = './uploads/test'
    os.makedirs(folder, exist_ok=True)
    item_path = ""
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

    # Check if the item is a folder
        if os.path.isdir(item_path):
            # Remove the folder and its contents
            shutil.rmtree(item_path)

    imagefolder = os.path.join(folder, 'image')
    clothfolder = os.path.join(folder, 'cloth-mask')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'agnostic-mask')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'agnostic-v3.2')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'gt_cloth_warped_mask')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'image-densepose')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'image-parse-agnostic-v3.2')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'openpose_img')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'openpose_json')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'image-parse-v3')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'output')
    os.makedirs(clothfolder, exist_ok=True)
    clothfolder = os.path.join(folder, 'cloth')

    os.makedirs(imagefolder, exist_ok=True)
    os.makedirs(clothfolder, exist_ok=True)

    if user_photo and selected_cloth:
        user_photo_path = os.path.join(imagefolder, user_photo.filename)
        user_photo.save(user_photo_path)
        
        # Resize the image to 768x1024
        resized_path = user_photo_path
        with Image.open(user_photo_path) as img:
            img = img.resize((768, 1024), Image.LANCZOS)  # Resize with high-quality downsampling
            img.save(resized_path)

        
        source_path = CLOTH_FOLDER + "/" + selected_cloth
        

        # Target path
        target_path = clothfolder + "/" + selected_cloth

        # Copy file
        shutil.copy(source_path, target_path)

        print("starting the preprocessing")
        
    
        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/openposetest.py"'

        

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        print("starting the preprocessing first done")


        empty_files = []
        directory_path = "./uploads/test/openpose_json"
    
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):  # Ensure we only process JSON files
                file_path = os.path.join(directory_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        
                        if isinstance(data, dict) and data.get("people") == []:
                            empty_files.append(filename)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {filename}: {e}")

        hh = 0
        for filename in empty_files:
            filepath = os.path.join("./uploads/test/openpose_json", filename)
            os.remove(filepath)
            empty_files[hh] = filename[:-14] + 'rendered.png'
            hh+=1
        
        hh=0

        for filename in empty_files:
            filepath = os.path.join("./uploads/test/openpose_img", filename)
            os.remove(filepath)
            empty_files[hh] = filename[:-12] + 'person.jpg'
            hh+=1

        
        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/denseposetest.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()
        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/clothmaskv2test.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageresizertest.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageparsev3test.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        if process.returncode != 0:
            print(f"\nError: Process exited with code {process.returncode}")
        else:
            print("\nCommand executed successfully")

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/zerorderholdtest.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/imageparseagnosticv3.2test.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/agnosticv3.2test.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/agnosticmasktest.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        command = f'conda run -n test python3 -u "{PREPROCESSING_FOLDER}/gtwarpedclothmasktest.py"'

        print("Running command:", command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()


        files_to_send = []
        folder_names = ['agnostic-mask', 'agnostic-v3.2', 'cloth', 'cloth-mask', 'gt_cloth_warped_mask', 'image', 'image-densepose', 'image-parse-agnostic-v3.2', 'image-parse-v3', 'openpose_img', 'openpose_json']
        folder_path = folder
        print(folder)
        file_name = user_photo.filename[:-4]
        file_tuple = []  # Collect matching files from all 11 folders
        for i in range(0, 11):  # Assuming folders are named 1 to 11
            folder_path = os.path.join(folder, folder_names[i])
            if i == 0:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 1:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 2:
                file_path = os.path.join(folder_path, selected_cloth[:-4] + '.jpg')
            if i == 3:
                file_path = os.path.join(folder_path, selected_cloth[:-4] + '.jpg')
            if i == 4:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 5:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 6:
                file_path = os.path.join(folder_path, file_name + '.jpg')
            if i == 7:
                file_path = os.path.join(folder_path, file_name + '.png')
            if i == 8:
                file_path = os.path.join(folder_path, file_name + '.png')
            if i == 9:
                file_path = os.path.join(folder_path, file_name + '_rendered.png')
            if i == 10:
                file_path = os.path.join(folder_path, file_name + '_keypoints.json')
            print(file_path)
            if os.path.exists(file_path):
                    file_tuple.append(('file', open(file_path, 'rb')))
            if len(file_tuple) == 11:  # Only send tuples of 11 files
                files_to_send.extend(file_tuple)

        print(files_to_send)

        # Send request to remote server
        # try:
        #     response = requests.post(
        #         REMOTE_SERVER_URL_TRY_ON,
        #         files=files_to_send,
        #         timeout=3600
        #     )

        #     user_photo.seek(0)  # Reset the file pointer for reuse
        #     user_image_data = user_photo.read()

        #     if response.status_code == 200:
        #         # Return both user image and try-on result as base64
        #         return jsonify({
        #             "message": "Try-on successful",
        #             "user_image": user_image_data.decode('latin1'),
        #             "try_on_result": response.content.decode('latin1')
        #         })
        #     else:
        #         return jsonify({
        #             "message": "Failed to generate try-on result.",
        #             "error": response.text
        #         }), response.status_code
        # finally:
        #     user_photo.close()



        # Close all opened file handles

        i = 0
        print()
        print()
        print()
        print(request.files)

        clothname=""
        imagename=""
        
        # for file in files:
        #     if(i==2):
        #         clothname=file.filename
        #     if(i==5):
        #         imagename=file.filename
        
        #     print(file)
        #     folder_name = folder_names[i % 11]
        #     file_name = file.filename

        #     # Ensure folder exists
        #     folder_path = os.path.join(upload, folder_name)
        #     os.makedirs(folder_path, exist_ok=True)

        #     # Save fil
        #     file_path = os.path.join(folder_path, file_name)
        #     file.save(file_path)
        #     i+=1
            # Maintain folder structure for response
        for file_tuple in files_to_send:
                
            if file_tuple:

                folder_name = folder_names[i % 11]
                file_object = file_tuple[1]  # Extract the file object
                file_name = os.path.basename(file_object.name)  # Get the correct filename
                if(i==2):
                    clothname=file_name
                if(i==5):
                    imagename=file_name
                # Ensure folder exists
                folder_path = os.path.join('./uploaded_files/test', folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Save file properly
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_object.read())  # Write file contents to disk

                print(f"Saved {file_name} to {folder_path}")

            i += 1
        
        test_file_path = os.path.join('./uploaded_files', 'test_pairs.txt')

        with open(test_file_path, 'w') as test_file:
            # Write filenames to train and test files
            
            test_file.write(f"{imagename} {clothname}\n")

        print(f"Test file created at: {test_file_path}")

        command = f'CUDA_VISIBLE_DEVICES=0 conda run -n StableVITON python ./StableVITON/inference.py --config_path ./StableVITON/configs/VITONHD.yaml --batch_size 1 --model_load_path {CKPT_PATH} --unpair'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream stdout/stderr
        for line in process.stdout:
            print(line, end="")
        for line in process.stderr:
            print(line, end="")
        process.wait()

        imagepath="./samples/unpair/" + imagename[:-4] + "_" + clothname[:-4] + ".jpg"
        
        
        with open(imagepath, "rb") as src_file:
            image_data = src_file.read()

        output_path = f"./uploads/test/output/{imagename[:-4]}.jpg"

        with open(output_path, "wb") as dest_file:
            dest_file.write(image_data)

    
        clothes = os.listdir(CLOTH_FOLDER)  # List all files in clothes folder
        return render_template('try_on.html', clothes=clothes, user_image_path=imagename[:-4] +'.jpg', result_image_path=imagename[:-4] +'.jpg')
        
    return redirect(url_for('landing_page'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
