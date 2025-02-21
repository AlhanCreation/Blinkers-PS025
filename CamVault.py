import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog,messagebox
import threading
from threading import Thread 
from playsound import playsound
import google.generativeai as genai
import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
from PIL import Image, ImageTk
import mediapipe as mp
import requests
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from SIH2 import mp_draw
from twilio.rest import Client
from datetime import datetime, timedelta
import torch

# Dictionary to track the last detection time for each person
last_detected_time = {}
video_capture = None

screenshot_counter = 1  # Initialize a global counter for naming screenshots
last_screenshot_time = {}  # Dictionary to track the last screenshot time for each person
screenshot_folder = "screenshots"  # Directory for storing screenshots
# Create the screenshots folder if it doesn't exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)
    
    
# Twilio credentials
TWILIO_ACCOUNT_SID = "AC58d0b9023a29485aa09a8c627f18bdaa"
TWILIO_AUTH_TOKEN = "6e2ca905067a05ca49cd2489b6808285"
TWILIO_PHONE = "+14582321954"
ALERT_PHONE = "+919673174369"
TWILIO_SMS = "+14582321954"
ALERT_SMS = "+919673174369"

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

cloudinary.config(
    cloud_name="dcvsavnx3",
    api_key="399294449426652",
    api_secret="7ZJuKbNHl0KsWPMAcKb4OWDH-j8"
)


# Mailgun API Credentials
MAILGUN_DOMAIN = "sandbox7d5a6b356d7c45ceaf0aa0daea2fc199.mailgun.org"
MAILGUN_API_KEY = "b4ca514bb6258ab1319dabf8222baaed-ac3d5f74-3a2e422b"  # Your actual API key
SENDER_EMAIL = f"mailgun@{MAILGUN_DOMAIN}"
RECIPIENT_EMAIL = "harshalborkar501@gmail.com"

def send_email_with_image(image_path,subject="⚠️ Suspicious Activity Detected!", message="Suspicious activity detected. See the attached image."):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            files={"attachment": ("screenshot.jpg", image_file, "image/jpeg")},
            data={
                "from": f"Mailgun Alerts <{SENDER_EMAIL}>",
                "to": RECIPIENT_EMAIL,
                "subject": "⚠️ Suspicious Activity Detected!",
                "text": "Suspicious activity detected by the surveillance system. See attached image."
            }
        )
    if response.status_code == 200:
        print("✅ Email with image sent successfully!")
    else:
        print(f"❌ Email failed! Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

# MongoDB credentials
MONGO_URI = "mongodb+srv://alhanmsiddique:jU0FQ5M89o3hlL1w@cluster0.o7nko.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "test"
COLLECTION_NAME = "listings"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
documents=collection.find()

for doc in documents:
    Title=doc.get("title")
    Height=doc.get("height")
    Weight=doc.get("weight")
    Occupations=doc.get("occupations")
    Description=doc.get("description")
    Contact=doc.get("price")
    Location="Priyadarshini College of Engineering,Nagpur"
    Country=doc.get("country")
    Reviews=doc.get("reviews")

stop_processing_flag = False
last_detected_time = {}  # Dictionary to store the last detection time for each person

# Global variables
known_face_encodings = []
known_face_names = []
dangerous_objects = ["knife", "gun", "bomb"]
stop_processing_flag = False
video_processing_running = False
recognized_faces = {}
activity_log = []


# Global variable to keep track of the last screenshot ID
last_screenshot_id = 0
screenshot_folder = "screenshots"

# Create a directory for screenshots if it does not exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def load_known_faces():
    global known_face_encodings, known_face_names
    print("Loading known faces...")
    face_folder = "faces_folder"
    for filename in os.listdir(face_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(face_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(filename.split(".")[0])
                print(f"Loaded {filename.split('.')[0]} from {image_path}")
            else:
                print(f"No face found in {image_path}. Skipping.")
# Create the screenshots folder if it doesn't exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)
    
def take_screenshot(label, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/{label}_{timestamp}.jpg"
    
    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite(filename, frame)
    
    print(f"Screenshot saved: {filename}")
    
    return filename

    # Save the screenshot with a sequential number
    screenshot_path = os.path.join(screenshot_folder, f"{screenshot_counter}.jpg")
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved: {screenshot_path}")

    # Update the last screenshot time and increment the counter
    last_screenshot_time[name] = current_time
    screenshot_counter += 1    

   
# Function to play alert sound
def play_alert_sound():
    playsound("alert.mp3", block=False) 
    
# Log recognized faces and objects to CSV
def log_to_csv(filename, data):
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    file_path = os.path.join(log_directory, filename)

    try:
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f"Logged data to {file_path}: {data}")
    except Exception as e:
        print(f"Error logging data to {file_path}: {e}")

        
# Start surveillance
def start_surveillance():
    global stop_processing_flag, video_capture
    stop_processing_flag = False
    print("Surveillance started!")

    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)  # Ensure only one capture object is active
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set capture width
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set capture height

    model = YOLO('yolov5n.pt')  #Use the lightweight YOLO model
    class_names = model.names
    frame_skip = 1  # Process every 2nd frame
    frame_count = 0
    while not stop_processing_flag:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        if frame_count % frame_skip == 0:
            process_frame(frame, model, class_names)  # Ensure this function doesn't open a new window

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    
def manage_dangerous_objects():
    def add_object():
        new_object = simpledialog.askstring("Add Object", "Enter the name of the new dangerous object:")
        if new_object and new_object not in dangerous_objects:
            dangerous_objects.append(new_object)
            messagebox.showinfo("Success", f"{new_object} added to dangerous objects.")

    def remove_object():
        object_to_remove = simpledialog.askstring("Remove Object", "Enter the name of the object to remove:")
        if object_to_remove in dangerous_objects:
            dangerous_objects.remove(object_to_remove)
            messagebox.showinfo("Success", f"{object_to_remove} removed from dangerous objects.")
        else:
            messagebox.showwarning("Not Found", f"{object_to_remove} not found in dangerous objects.")

    def view_objects():
        objects_list = "\n".join(dangerous_objects)
        messagebox.showinfo("Dangerous Objects", f"Current dangerous objects:\n{objects_list}")

    manage_window = tk.Toplevel()
    manage_window.title("Manage Dangerous Objects")
    manage_window.geometry("400x300")

    tk.Button(manage_window, text="Add Object", font=("Arial", 14), command=add_object).pack(pady=10)
    tk.Button(manage_window, text="Remove Object", font=("Arial", 14), command=remove_object).pack(pady=10)
    tk.Button(manage_window, text="View Objects", font=("Arial", 14), command=view_objects).pack(pady=10)


# Process a single frame for both face recognition and object detection
def process_frame(frame, model, class_names):
    global known_face_encodings, known_face_names

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Recognition
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[best_match_index]

            # Take a screenshot of the detected person
            # take_screenshot(name, frame)

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Object Detection
        results = model.predict(frame, conf=0.5)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                class_name = class_names[int(box.cls[0])]

                label = f"{class_name} ({confidence:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow("Surveillance", frame)

    except Exception as e:
        print(f"Error processing frame: {e}")
        
    # Object Detection
    results = model.predict(frame, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown ({class_id})"

            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if class_name in dangerous_objects:
                cv2.putText(frame, "DANGER!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                play_alert_sound()
                print(f"⚠️ {class_name} detected! Sending alert...")
                image_path = take_screenshot(class_name, frame)
                send_email_with_image(image_path)
                # image_path = take_screenshot(name, frame)
                send_sms_alert1(class_name)
                send_email_with_image(image_path, f"Dangerous Object Detected: {class_name}!",
                                          f"Surveillance detected a {class_name} in the monitored area.")
                log_to_csv('recognized_objects.csv', [class_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
     
     # Behavior Analysis with MediaPipe Pose
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        # Example behavior detection: Checking if hands are above the head
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            cv2.putText(frame, "Hands above head!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            log_to_csv('behavior_log.csv', ["Hands above head detected", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
     # Behavior detection: Checking if hands are hiding the face
        left_wrist_x = left_wrist.x * frame.shape[1]
        left_wrist_y = left_wrist.y * frame.shape[0]
        right_wrist_x = right_wrist.x * frame.shape[1]
        right_wrist_y = right_wrist.y * frame.shape[0]

        face_center_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x +
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x +
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x) / 3 * frame.shape[1]

        face_center_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y +
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y +
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y) / 3 * frame.shape[0]

        if (abs(left_wrist_x - face_center_x) < 100 and abs(left_wrist_y - face_center_y) < 100) or \
           (abs(right_wrist_x - face_center_x) < 100 and abs(right_wrist_y - face_center_y) < 100):
            cv2.putText(frame, "Suspicious", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            image_path = take_screenshot("Suspicious Activity", frame)
            send_alert("Suspicious activity detected! (Face hiding)", image_path)
            log_to_csv('behavior_log.csv', ["Face hiding detected", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    cv2.imshow("Surveillance", frame)
    
def send_alert(message, image_path=None):
    try:
        if image_path:
            # Send MMS with the screenshot
            message = twilio_client.messages.create(
                body=message,
                from_=+12346572562,
                to=+919158407565,
                media_url=[f"file://{os.path.abspath(image_path)}"]  # Ensure the file is accessible
            )
        else:
            # Send SMS alert
            message = twilio_client.messages.create(
                body=message,
                from_=+12346572562,
                to=+919158407565
            )
        
        print(f"Alert sent: {message.sid}")
    
    except Exception as e:
        print(f"Failed to send alert: {e}")

   
def send_sms_alert1(name):
    message_body =f"Alert: {name} : Dangerous object detected"
    twilio_client.messages.create(
        body=message_body,
        from_=TWILIO_SMS,
        to=ALERT_SMS
    )
def send_sms_alert(name):
    message_body =( f"Alert: {name} has been identified by the surveillance system."
        f"📏 *Height*: {Height}\n"
        f"💼 *Occupations*:{Occupations}\n"
        f"📍 *Location*: {Location}\n"
        f"🌍 *Country*: {Country}\n"
        f"💰 *Contact*: {Contact}\n"
        )
    twilio_client.messages.create(
        body=message_body,
        from_=TWILIO_SMS,
        to=ALERT_SMS
    )
    print(f"SMS alert sent about {name}")

def send_whatsapp_alert(suspect_details):
    # Format message with details about the suspect
    message_content = (
        f"🚨 *ALERT: Suspect Detected!* 🚨\n"
        f"👤 *Title*: {suspect_details.get('title', 'Unknown')}\n"
        f"📏 *Height*: {Height}\n"
        f"💼 *Occupations*:{Occupations}\n"
        f"📍 *Location*: {Location}\n"
        f"🌍 *Country*: {Country}\n"
        f"💰 *Contact*: {Contact}\n"
        f"📷 *Image URLs*: {', '.join(suspect_details.get('image_urls', ['None']))}\n"
    )

    # Send WhatsApp message using Twilio
    twilio_client.messages.create(
        body=message_content,
        from_='whatsapp:' + TWILIO_PHONE,
        to='whatsapp:' + ALERT_PHONE
    )
    print(f"WhatsApp alert sent with details about {suspect_details.get('title', 'Unknown')}")

    
# Stop surveillance
def stop_processing():
    global stop_processing_flag
    stop_processing_flag = True
    print("Surveillance will stop shortly!")


# Add face to system
def add_face():
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    if name:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg")])
        if file_path:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)

                # Save the local file
                save_path = os.path.join("faces_folder", f"{name}.jpg")
                cv2.imwrite(save_path, cv2.imread(file_path))

                # Upload to Cloudinary
                try:
                    response = cloudinary.uploader.upload(file_path, folder="faces_folder")
                    cloudinary_url = response.get("secure_url")
                    print(f"Uploaded to Cloudinary: {cloudinary_url}")

                    # Log the upload to CSV
                    log_to_csv('activity_log.csv', [
                        f"Added face: {name} (Cloudinary URL: {cloudinary_url})",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ])
                    messagebox.showinfo("Success", f"Face for {name} uploaded successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to upload image: {e}")
            else:
                messagebox.showwarning("Warning", "No face found in the selected image.")

def start_surveillance_threaded():
    # Start surveillance in a separate thread to prevent GUI freezing
    surveillance_thread = Thread(target=start_surveillance)
    surveillance_thread.start()
    
# Manage objects
def manage_objects():
    action = simpledialog.askstring("Choose Action", "Do you want to 'add' or 'delete' a dangerous object?")
    if action:
        action = action.lower()
        if action == "add":
            dangerous_object = simpledialog.askstring("Input", "Enter a dangerous object to log (e.g., knife, gun, bomb):")
            if dangerous_object and dangerous_object not in dangerous_objects:
                dangerous_objects.append(dangerous_object)
                log_to_csv('activity_log.csv', [f"Added dangerous object: {dangerous_object}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                print(f"Added dangerous object: {dangerous_object}")
            elif dangerous_object in dangerous_objects:
                print(f"{dangerous_object} is already in the list.")
        elif action == "delete":
            dangerous_object = simpledialog.askstring("Input", "Enter a dangerous object to delete:")
            if dangerous_object in dangerous_objects:
                dangerous_objects.remove(dangerous_object)
                log_to_csv('activity_log.csv', [f"Deleted dangerous object: {dangerous_object}", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                print(f"Deleted dangerous object: {dangerous_object}")
            else:
                print(f"{dangerous_object} not found in the list.")
        else:
            print("Invalid action! Please choose 'add' or 'delete'.")

# View activity log
def view_activity_log():
    log_window = tk.Toplevel()
    log_window.title("Activity Log")
    log_text = tk.Text(log_window, height=20, width=80)
    log_text.pack(padx=10, pady=10)

    # Read the log file and display its contents
    try:
        with open("activity_log.csv", "r") as log_file:
            log_text.insert(tk.END, log_file.read())
    except FileNotFoundError:
        log_text.insert(tk.END, "No activity log found.")
    
    log_text.config(state=tk.DISABLED)
    
    
# Process a recorded video
def process_recorded_video():
    global recorded_video_processing_flag
    recorded_video_processing_flag = True  # Start processing

    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        print("No video file selected.")
        recorded_video_processing_flag = False  # Reset the flag if no file selected
        return

    def process_video():
        global recorded_video_processing_flag

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file.")
            recorded_video_processing_flag = False
            return

        model = YOLO('yolov5n.pt')  # Initialize model
        model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Set device

        class_names = model.names
        frame_skip = 2  # Process every 2nd frame for speed
        frame_count = 0

        while cap.isOpened() and recorded_video_processing_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video or failed to read frame.")
                break

            if frame_count % frame_skip == 0:
                resized_frame = cv2.resize(frame, (640, 640))  # Resize for faster inference
                processed_frame = process_frame(resized_frame, model, class_names)
                if processed_frame is not None:
                    cv2.imshow("Processed Video", processed_frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow manual exit via 'q'
                break

        cap.release()
        cv2.destroyAllWindows()
        recorded_video_processing_flag = False  # Reset the flag
    threading.Thread(target=process_video, daemon=True).start()

# Stop recorded video processing
def stop_recorded_video_processing():
    global recorded_video_processing_flag
    recorded_video_processing_flag = False  # Set the flag to False
    print("Stop signal sent for recorded video processing.")
                
def load_activity_log(text_widget):
    log_path = os.path.join("logs", "recognized_faces.csv")  # Ensure correct path
    if os.path.exists(log_path):
        with open(log_path, "r") as log_file:
            content = log_file.read()
            text_widget.config(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
    else:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, "No activity log found.")
        text_widget.config(state=tk.DISABLED)

def create_control_bar(root):
    """Create a custom control bar with Minimize and Exit buttons."""
    control_bar = tk.Frame(root, bg="gray", height=30)
    control_bar.pack(side=tk.TOP, fill=tk.X)

    # Exit button
    exit_button = tk.Button(control_bar, text="X", bg="red", fg="white", font=("Arial", 12, "bold"), command=root.destroy)
    exit_button.pack(side=tk.RIGHT, padx=5, pady=2)

    # Minimize button
    minimize_button = tk.Button(control_bar, text="_", bg="blue", fg="white", font=("Arial", 12, "bold"), command=lambda: root.iconify())
    minimize_button.pack(side=tk.RIGHT, padx=5, pady=2)

    # Title label
    title_label = tk.Label(control_bar, text="Surveillance System", bg="gray", fg="white", font=("Arial", 14, "bold"))
    title_label.pack(side=tk.LEFT, padx=10)


def create_gui():
    root = tk.Tk()
    root.title("Surveillance System")
    # root.geometry("1420x1080")
    root.attributes('-fullscreen', True) 
    root.bind("<Escape>", lambda event: root.destroy())
    
    create_control_bar(root)
    
    # Header with Blinkers logo
    header = tk.Frame(root, bg="white", height=100)
    header.pack(fill=tk.X, side=tk.TOP)

    logo_path = "blinkers_logo.png"  # Path to the Blinkers logo
    logo_image = Image.open(logo_path)
    logo_image = logo_image.resize((80, 80), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(header, image=logo_photo, bg="white")
    logo_label.image = logo_photo
    logo_label.pack(side=tk.LEFT, padx=10, pady=10)

    header_title = tk.Label(header, text="Face First", font=("Arial", 24, "bold"), bg="white")
    header_title.pack(side=tk.LEFT, padx=20)
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH)

    # Tab 1: Real-Time Surveillance
    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Real-Time Surveillance")

    feed_label = tk.Label(tab1, bg="black")
    feed_label.pack(expand=True, fill=tk.BOTH)

    tab1_buttons_frame = tk.Frame(tab1)
    tab1_buttons_frame.pack(side=tk.BOTTOM, pady=20)

    tk.Button(tab1_buttons_frame, text="Start Surveillance", command=start_surveillance_threaded).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Stop Surveillance", command=stop_processing).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Add Face", command=add_face).pack(side=tk.LEFT, padx=10)
    tk.Button(tab1_buttons_frame, text="Manage Dangerous Objects", command=manage_dangerous_objects).pack(side=tk.LEFT, padx=10)
    tab1_buttons_frame = tk.Frame(tab1)
    tab1_buttons_frame.pack(side=tk.BOTTOM, pady=20)

    tab2 = tk.Frame(notebook)
    notebook.add(tab2, text="Recorded Video Processing")

    tk.Button(tab2, text="Process Video", command=process_recorded_video).pack(pady=20)
    tk.Button(tab2, text="Stop Video Processing", command=stop_recorded_video_processing).pack(pady=10)
    
    # Tab 3: View Activity Log
    tab3 = tk.Frame(notebook)
    notebook.add(tab3, text="View Activity Log")

    activity_log_text = tk.Text(tab3, state=tk.DISABLED, wrap=tk.WORD)
    activity_log_text.pack(expand=True, fill=tk.BOTH)

    tk.Button(tab3, text="Load Activity Log", command=lambda: load_activity_log(activity_log_text)).pack(pady=10)
    root.mainloop()

# Start the program
if __name__ == "__main__":
    load_known_faces()
    create_gui()