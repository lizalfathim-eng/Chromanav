import os
import queue
import re
import threading
import time

import cv2
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO


MODEL_PATH = os.environ.get(
    "CHROMANAV_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"),
)
CONFIDENCE_THRESHOLDS = {
    "Traffic Light": 0.5,
    "Stop": 0.5,
}
INFERENCE_SIZE = 640
MAX_INFERENCE_FPS = 8
ANNOUNCEMENT_COOLDOWN = 5.0
TRAFFIC_LIGHT_COOLDOWN = 3.0

SUPPORTED_DETECTION_LABELS = {
    "traffic light": "Traffic Light",
    "stop sign": "Stop",
}

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
SIGN_COLOR = (255, 180, 0)
WHITE = (255, 255, 255)


def normalize_class_name(class_name):
    normalized = re.sub(r"[_-]+", " ", class_name.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def canonical_detection_label(class_name):
    """Return a display label only for classes supported by the bundled model."""
    return SUPPORTED_DETECTION_LABELS.get(normalize_class_name(class_name))


def detect_traffic_light_color(crop_img):
    if crop_img is None or crop_img.size == 0:
        return None

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    masks = {
        "Red Light": (
            cv2.inRange(hsv, np.array((0, 100, 80)), np.array((10, 255, 255))),
            cv2.inRange(hsv, np.array((170, 100, 80)), np.array((180, 255, 255))),
        ),
        "Yellow Light": (
            cv2.inRange(hsv, np.array((15, 100, 80)), np.array((35, 255, 255))),
        ),
        "Green Light": (
            cv2.inRange(hsv, np.array((36, 80, 60)), np.array((90, 255, 255))),
        ),
    }

    height, width = crop_img.shape[:2]
    thirds = (
        (0, height // 3),
        (height // 3, 2 * height // 3),
        (2 * height // 3, height),
    )
    expected_positions = {
        "Red Light": thirds[0],
        "Yellow Light": thirds[1],
        "Green Light": thirds[2],
    }

    position_scores = {}
    full_counts = {}
    for color, color_masks in masks.items():
        merged_mask = color_masks[0].copy()
        for mask in color_masks[1:]:
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        full_counts[color] = cv2.countNonZero(merged_mask)

        top, bottom = expected_positions[color]
        position_scores[color] = cv2.countNonZero(merged_mask[top:bottom, :])

    if height >= 9:
        color, count = max(position_scores.items(), key=lambda item: item[1])
        zone_area = width * max(1, height // 3)
        if count >= max(2, int(zone_area * 0.005)):
            return color

    color, count = max(full_counts.items(), key=lambda item: item[1])
    return color if count >= max(3, int(height * width * 0.005)) else None


def class_name_for(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, ""))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return ""


def run_detection(stop_event, speech_queue, status_queue):
    camera = None
    failed = False
    try:
        status_queue.put("Loading detection model...")
        model = YOLO(MODEL_PATH)
        status_queue.put(f"Model loaded: {os.path.basename(MODEL_PATH)}")
        print(f"Model loaded: {MODEL_PATH}")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Could not open the webcam.")

        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        status_queue.put("Detecting. Press q in the video window to quit.")
        last_announced = {}
        minimum_frame_time = 1.0 / MAX_INFERENCE_FPS
        while not stop_event.is_set():
            frame_started = time.monotonic()
            captured, frame = camera.read()
            if not captured:
                raise RuntimeError("Failed to capture a frame from the webcam.")

            results = model(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=min(CONFIDENCE_THRESHOLDS.values()),
                verbose=False,
            )
            detected_labels = set()

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_name_for(model, class_id)
                    label = canonical_detection_label(class_name)
                    if label is None or confidence < CONFIDENCE_THRESHOLDS[label]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    height, width = frame.shape[:2]
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    if label == "Traffic Light":
                        light_color = detect_traffic_light_color(frame[y1:y2, x1:x2])
                        label = light_color or label

                    detected_labels.add(label)
                    color = (
                        RED if label == "Red Light"
                        else YELLOW if label == "Yellow Light"
                        else GREEN if label == "Green Light"
                        else SIGN_COLOR if label != "Traffic Light"
                        else WHITE
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.0%}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            now = time.monotonic()
            ready_to_announce = []
            for label in sorted(detected_labels):
                cooldown = (
                    TRAFFIC_LIGHT_COOLDOWN
                    if label in ("Red Light", "Yellow Light", "Green Light", "Traffic Light")
                    else ANNOUNCEMENT_COOLDOWN
                )
                if now - last_announced.get(label, float("-inf")) >= cooldown:
                    ready_to_announce.append(label)

            if ready_to_announce:
                message = ", ".join(ready_to_announce)
                print(f"Detected: {message}")
                status_queue.put(f"Detected: {message} - sending to voice")
                speech_queue.put((10, time.monotonic_ns(), message))
                for label in ready_to_announce:
                    last_announced[label] = now

            cv2.imshow("ChromaNav - Traffic Sign & Light Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
            remaining_frame_time = minimum_frame_time - (time.monotonic() - frame_started)
            if remaining_frame_time > 0:
                time.sleep(remaining_frame_time)
    except Exception as error:
        failed = True
        status_queue.put(f"Detection error: {error}")
        print(f"Detection error: {error}")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if not failed:
            status_queue.put("Status: Stopped")


def speech_worker(speech_queue, voice_status_queue):
    try:
        engine = pyttsx3.init("sapi5") if os.name == "nt" else pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        voice_status_queue.put("Speech engine initialized")
    except Exception as error:
        voice_status_queue.put(f"Voice output unavailable: {error}")
        print(f"Voice output unavailable: {error}")
        return

    while True:
        _, _, message = speech_queue.get()
        if message is None:
            return
        try:
            voice_status_queue.put(f"Speaking: {message}")
            engine.say(message)
            engine.runAndWait()
            voice_status_queue.put(f"Spoken: {message}")
        except Exception as error:
            voice_status_queue.put(f"Voice output error: {error}")
            print(f"Voice output error: {error}")
            return


def main():
    root = tk.Tk()
    root.title("ChromaNav Traffic Sign & Light Detection")
    root.geometry("500x320")
    root.configure(bg="#2E2E2E")

    status_queue = queue.Queue()
    voice_status_queue = queue.Queue()
    speech_queue = queue.PriorityQueue()
    stop_event = threading.Event()
    detection_thread = None
    threading.Thread(
        target=speech_worker,
        args=(speech_queue, voice_status_queue),
        daemon=True,
    ).start()

    heading_label = tk.Label(
        root,
        text="ChromaNav",
        font=("Helvetica", 28, "bold"),
        bg="#2E2E2E",
        fg="#FFFFFF",
    )
    heading_label.pack(pady=30)

    button_frame = tk.Frame(root, bg="#2E2E2E")
    button_frame.pack(pady=20)
    status_label = tk.Label(
        root,
        text="Status: Ready",
        font=("Helvetica", 12),
        bg="#2E2E2E",
        fg="#FFFFFF",
    )
    status_label.pack(pady=10)
    voice_status_label = tk.Label(
        root,
        text="Voice: Starting...",
        font=("Helvetica", 10),
        bg="#2E2E2E",
        fg="#FFFFFF",
    )
    voice_status_label.pack(pady=4)

    def start_detection():
        nonlocal detection_thread
        if detection_thread is not None and detection_thread.is_alive():
            return
        stop_event.clear()
        detection_thread = threading.Thread(
            target=run_detection,
            args=(stop_event, speech_queue, status_queue),
            daemon=True,
        )
        detection_thread.start()
        status_label.config(text="Status: Starting...")
        start_button.config(state=tk.DISABLED)

    def stop_detection():
        stop_event.set()
        status_label.config(text="Status: Stopping...")

    def test_voice():
        speech_queue.put((0, time.monotonic_ns(), "ChromaNav voice output is working"))
        status_label.config(text="Status: Voice test queued - will speak next")

    def poll_status():
        while True:
            try:
                message = status_queue.get_nowait()
            except queue.Empty:
                break
            status_label.config(text=message)
        while True:
            try:
                message = voice_status_queue.get_nowait()
            except queue.Empty:
                break
            voice_status_label.config(text=f"Voice: {message}")
        if detection_thread is None or not detection_thread.is_alive():
            start_button.config(state=tk.NORMAL)
        root.after(200, poll_status)

    def close_app():
        stop_event.set()
        speech_queue.put((-1, time.monotonic_ns(), None))
        root.destroy()

    start_button = ttk.Button(
        button_frame,
        text="Start Detection",
        command=start_detection,
    )
    start_button.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=5)
    stop_button = ttk.Button(
        button_frame,
        text="Stop Detection",
        command=stop_detection,
    )
    stop_button.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=5)
    voice_button = ttk.Button(
        button_frame,
        text="Test Voice",
        command=test_voice,
    )
    voice_button.pack(side=tk.LEFT, padx=20, pady=10, ipadx=10, ipady=5)

    root.protocol("WM_DELETE_WINDOW", close_app)
    root.after(200, poll_status)
    root.mainloop()


if __name__ == "__main__":
    main()
