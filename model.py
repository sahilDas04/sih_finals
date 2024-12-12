import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Model Configuration
try:
    model = YOLO('yolov8n.pt')  # Using default model
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Thresholds for Anomalies
MAX_WAITING_TIME = 10  # seconds
MIN_BOX_SIZE = 0.01  # Minimum relative size of a bounding box
MAX_BOX_SIZE = 0.5   # Maximum relative size of a bounding box
CONFIDENCE_THRESHOLD = 0.5
TRACKING_THRESHOLD = 50  # Pixel distance for tracking same person

class PersonTracker:
    def __init__(self):
        self.tracked_persons = {}
        self.next_id = 0

    def track_person(self, bbox, current_time):
        closest_match = None
        min_distance = float('inf')

        for track_id, track_info in self.tracked_persons.items():
            last_bbox = track_info['last_bbox']

            current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            previous_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)

            distance = np.sqrt(
                (current_center[0] - previous_center[0])**2 + 
                (current_center[1] - previous_center[1])**2
            )

            if distance < min_distance and distance < TRACKING_THRESHOLD:
                min_distance = distance
                closest_match = track_id

        if closest_match is not None:
            track_id = closest_match
            self.tracked_persons[track_id]['last_bbox'] = bbox
            self.tracked_persons[track_id]['last_seen'] = current_time

            waiting_time = current_time - self.tracked_persons[track_id]['entry_time']
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = bbox[2] * bbox[3]
            relative_box_size = box_area / img_area

            if waiting_time > MAX_WAITING_TIME:
                self.tracked_persons[track_id]['anomalies'].append(
                    f"Waiting too long: {waiting_time:.2f} seconds"
                )

            if relative_box_size < MIN_BOX_SIZE or relative_box_size > MAX_BOX_SIZE:
                self.tracked_persons[track_id]['anomalies'].append(
                    f"Abnormal size detected: {relative_box_size:.4f}"
                )
        else:
            track_id = self.next_id
            self.tracked_persons[track_id] = {
                'entry_time': current_time,
                'last_seen': current_time,
                'last_bbox': bbox,
                'anomalies': [],
                'service_time': 0  # Initialize service time
            }
            self.next_id += 1

        return track_id

    def calculate_service_time(self, current_time):
        for track_id, track_info in self.tracked_persons.items():
            track_info['service_time'] = current_time - track_info['entry_time']

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    person_tracker = PersonTracker()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        if frame_count % (frame_rate * 2) == 0:
            results = model.predict(frame, verbose=False)

            current_time = frame_count / frame_rate

            for detection in results[0].boxes:
                conf = float(detection.conf[0])
                label = int(detection.cls[0])
                bbox = detection.xyxy[0].tolist()

                if label == 0 and conf > CONFIDENCE_THRESHOLD:
                    person_tracker.track_person(bbox, current_time)

            # Update service times for all persons
            person_tracker.calculate_service_time(current_time)

    cap.release()

    final_results = {
        'total_persons': len(person_tracker.tracked_persons),
        'person_details': {}
    }

    for person_id, person_data in person_tracker.tracked_persons.items():
        final_results['person_details'][f'person_{person_id}'] = {
            'entry_time': person_data['entry_time'],
            'last_seen': person_data['last_seen'],
            'service_time': person_data['service_time'],
            'anomalies': person_data['anomalies']
        }

    return final_results

# Streamlit App
st.title("Video Processing with YOLO")
st.write("Upload a video to process and analyze using YOLO.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = f"temp_{uploaded_file.name}"

    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("Processing video..."):
            result = process_video(video_path)

        st.success("Video processed successfully!")
        st.json(result)

        os.remove(video_path)  # Clean up temporary file
    except Exception as e:
        st.error(f"Error processing video: {e}")
