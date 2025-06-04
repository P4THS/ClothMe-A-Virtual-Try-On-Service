from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

instance_names = 1



@app.route('/')
def landingpage():
    return render_template('landingpage.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            brand_name TEXT,
            email TEXT,
            phone TEXT,
            contact_person TEXT,
            password TEXT,
            card_number TEXT,
            expiry_date TEXT,
            cvv TEXT,
            url TEXT
        )
    ''')
    conn.commit()
    conn.close()

import subprocess
import time

def create_gcp_instance():
    global instance_names
    """Function to create a new GCP instance, install NVIDIA driver via apt,
    run your container, and then connect via SSH to view startup logs."""
    
    command = f"""#!/bin/bash
    gcloud compute instances create fypinstance{instance_names} \
        --machine-type=n2-standard-8 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=200GB \
        --tags=http-server,https-server \
        --zone=asia-southeast1-a \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata startup-script='#!/bin/bash
          # Update package lists and install required packages
          sudo apt update && sudo apt install -y docker.io curl build-essential linux-headers-$(uname -r) software-properties-common
          
          
          # Start and enable Docker
          systemctl start docker
          systemctl enable docker
          
          # Install Google Cloud SDK and configure Docker authentication
          sudo apt install -y google-cloud-sdk
          gcloud auth configure-docker
          
          # Retrieve and load your Docker image
          sudo gsutil cp -Z gs://clothme-docker-bucket/fyp2_mideval_v2.tar /tmp/fyp2_mideval_v2.tar
          sudo docker load -i /tmp/fyp2_mideval_v2.tar
        '
    """
    
    # Execute the command to create the instance.
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Instance creation command executed.")
    
    # Wait to ensure the instance is up and ready for SSH connection.
    # print("Waiting for instance to be ready for SSH connection...")
    # time.sleep(30)  # Adjust this delay as necessary

    # # Connect via SSH and tail the system log so you can see the script output in real time.
    # print("Connecting to the instance via SSH...")
    # ssh_command = f"gcloud compute ssh fypinstance{instance_names} --zone=asia-southeast1-a --command 'sudo tail -f /var/log/syslog'"
    # subprocess.run(ssh_command, shell=True)
    
    # print("SSH session ended.")





def get_instance_external_ip(instance_name, zone="asia-southeast1-a"):
    """Fetch the external IP of a GCP instance using gcloud command."""
    command = f"gcloud compute instances describe {instance_name} --zone={zone} --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        return result.stdout.strip()  # Get the IP address
    else:
        print("Error fetching external IP:", result.stderr)
        return None




def send_email(user_email, instance_url):
    sender_email = ""
    sender_password = ""  # Use the App Password here

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = user_email
    msg["Subject"] = "Your Account is Registered!"
    msg.attach(MIMEText(f"Your application is now available at: {instance_url}", "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())
        print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route("/register", methods=["POST"])
def register():
    global instance_names

    company_name = request.form.get("companyName")
    brand_name = request.form.get("brandName")
    email = request.form.get("email")
    phone = request.form.get("phone")
    contact_person = request.form.get("contactPerson")
    password = request.form.get("password")
    card_number = request.form.get("cardNumber")
    expiry_date = request.form.get("expiryDate")
    cvv = request.form.get("cvv")

    create_gcp_instance()

    time.sleep(30)

    # Get external IP of the running GCP instance
    instance_ip = get_instance_external_ip(f"fypinstance{instance_names}")
    instance_url = f"http://{instance_ip}:5000"

    instance_names += 1

    # Store user details along with the instance URL in the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (company_name, brand_name, email, phone, contact_person, password, card_number, expiry_date, cvv, url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (company_name, brand_name, email, phone, contact_person, password, card_number, expiry_date, cvv, instance_url))
    conn.commit()
    conn.close()

    # Send the instance URL to the user via email
    send_email(email, instance_url)

    return render_template('landingpage.html')

if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=8000, debug=True)
