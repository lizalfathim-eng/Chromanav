import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from threading import Thread

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv8 pre-trained model
try:
    model = YOLO("yolov8n.pt")  # Ensure this file exists in the directory
    print("✅ YOLOv8 Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

# Define Indian traffic sign classes
traffic_sign_classes = {
    9: "Traffic Light",
    11: "Stop",
    12: "Speed Limit",
    13: "Pedestrian Crossing",
    14: "No Parking",
    15: "One Way",
    16: "Give Way",  # Equivalent to Yield Sign
    17: "School Zone",
    18: "U-Turn Prohibited",
    19: "Horn Prohibited",
    20: "Compulsory Left/Right",
    21: "Hospital Ahead",
    22: "Rest Area",
    23: "Toll Booth Ahead",
    24: "Road Work",
    25: "Slippery Road",
}

# Define RGB color constants
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)  # Used in Indian information signs
WHITE = (255, 255, 255)

# Define color mapping for Indian traffic signs
color_mapping = {
    "Stop": RED,
    "No Parking": RED,
    "One Way": RED,
    "Give Way": YELLOW,
    "Speed Limit": YELLOW,
    "School Zone": YELLOW,
    "Pedestrian Crossing": YELLOW,
    "U-Turn Prohibited": RED,
    "Horn Prohibited": RED,
    "Compulsory Left/Right": BLUE,
    "Hospital Ahead": BLUE,
    "Rest Area": BLUE,
    "Toll Booth Ahead": BLUE,
    "Road Work": YELLOW,
    "Slippery Road": YELLOW,
}

# HSV color ranges for traffic light detection
def detect_traffic_light_color(crop_img):
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Define color ranges (HSV)
    color_ranges = {
        "Red Light": [(0, 120, 70), (10, 255, 255)],
        "Yellow Light": [(20, 100, 100), (30, 255, 255)],
        "Green Light": [(40, 100, 100), (90, 255, 255)]
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 50:  # Threshold to confirm color presence
            return color_name

    return None

# Function to start detection
def start_detection():
    global running
    running = True
    cap = cv2.VideoCapture(0)  # 0 = Default webcam

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture video")
            break

        # Run YOLO object detection
        results = model(frame)

        detected_objects = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # Class ID
                conf = float(box.conf[0])  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

                # Traffic light detection (color analysis)
                if cls == 9 and conf > 0.4:  # Traffic light ID
                    traffic_light_crop = frame[y1:y2, x1:x2]  # Crop detected traffic light
                    light_color = detect_traffic_light_color(traffic_light_crop)
                    
                    if light_color:
                        detected_objects.append(light_color)
                        label = light_color
                        color = RED if "Red" in light_color else GREEN if "Green" in light_color else YELLOW
                    else:
                        label = "Traffic Light"
                        color = WHITE  # Unknown color
                
                # Traffic signs detection
                elif cls in traffic_sign_classes and conf > 0.4:
                    label = traffic_sign_classes[cls]
                    detected_objects.append(label)

                    # Get color from dictionary, default to green
                    color = color_mapping.get(label, GREEN)

                else:
                    continue

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Voice feedback for detected signs and traffic light colors
        if detected_objects:
            message = f"Detected: {', '.join(detected_objects)}"
            print(message)
            engine.say(message)
            engine.runAndWait()

        # Show video feed
        cv2.imshow("Indian Traffic Sign & Light Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to stop detection
def stop_detection():
    global running
    running = False

# Create the main window
root = tk.Tk()
root.title("Traffic Sign & Light Detection")
root.geometry("500x400")
root.configure(bg="#2E2E2E")  # Dark background color

# Add a heading label
heading_label = tk.Label(root, text="ChromaNav", font=("Helvetica", 28, "bold"), bg="#2E2E2E", fg="#FFFFFF")
heading_label.pack(pady=20)

# Create a frame for buttons
button_frame = tk.Frame(root, bg="#2E2E2E")
button_frame.pack(pady=20)

# Create Start Detection button
start_button = ttk.Button(button_frame, text="Start Detection", command=lambda: Thread(target=start_detection).start())
start_button.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=5)

# Create Stop Detection button
stop_button = ttk.Button(button_frame, text="Stop Detection", command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=5)

# Add a status label
status_label = tk.Label(root, text="Status: Ready", font=("Helvetica", 12), bg="#2E2E2E", fg="#FFFFFF")
status_label.pack(pady=10)

# Initialize running variable
running = False

# Start the GUI event loop
root.mainloop()