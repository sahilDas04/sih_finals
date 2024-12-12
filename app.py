from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from flask_cors import cross_origin
import time

app = Flask(__name__)


# Model Configuration
try:
    model = YOLO('yolov8n.pt')  # Using default model
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Thresholds for Anomalies
MAX_WAITING_TIME = 20  # seconds
MIN_BOX_SIZE = 0.01  # Minimum relative size of a bounding box
MAX_BOX_SIZE = 0.5   # Maximum relative size of a bounding box
CONFIDENCE_THRESHOLD = 0.5
TRACKING_THRESHOLD = 50  # Pixel distance for tracking same person

class PersonTracker:
    def __init__(self):
        self.tracked_persons = {}
        self.next_id = 0

    def track_person(self, bbox, current_time):
        # Find closest existing track
        closest_match = None
        min_distance = float('inf')
        
        for track_id, track_info in self.tracked_persons.items():
            last_bbox = track_info['last_bbox']
            
            # Calculate center of current and previous bounding boxes
            current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            previous_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
            
            # Calculate Euclidean distance
            distance = np.sqrt(
                (current_center[0] - previous_center[0])**2 + 
                (current_center[1] - previous_center[1])**2
            )
            
            if distance < min_distance and distance < TRACKING_THRESHOLD:
                min_distance = distance
                closest_match = track_id
        
        # Assign or create new track
        if closest_match is not None:
            track_id = closest_match
            self.tracked_persons[track_id]['last_bbox'] = bbox
            self.tracked_persons[track_id]['last_seen'] = current_time
            
            # Update anomalies
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
            # New person detected
            track_id = self.next_id
            self.tracked_persons[track_id] = {
                'entry_time': current_time,
                'last_seen': current_time,
                'last_bbox': bbox,
                'anomalies': []
            }
            self.next_id += 1
        
        return track_id
    
    # def calculate_service_time(self, current_time):
    #     for track_id, track_info in self.tracked_persons.items():
    #         track_info['service_time'] = current_time - track_info['entry_time']

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    
    # Video metadata
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracker
    person_tracker = PersonTracker()
    
    # Processing variables
    frame_count = 0
    processed_frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Process every 2 seconds
        if frame_count % (frame_rate * 2) == 0:
            # Run YOLO detection
            results = model.predict(frame, verbose=False)
            
            current_time = frame_count / frame_rate
            
            # Process detections
            for detection in results[0].boxes:
                # Get detection details
                conf = float(detection.conf[0])
                label = int(detection.cls[0])
                bbox = detection.xyxy[0].tolist()
                
                # Only process person detections
                if label == 0 and conf > CONFIDENCE_THRESHOLD:
                    # Track person and update anomalies
                    person_id = person_tracker.track_person(bbox, current_time)

    cap.release()

    # Prepare final results
    final_results = {
        'total_persons': len(person_tracker.tracked_persons),
        'person_details': {}
    }

    for person_id, person_data in person_tracker.tracked_persons.items():
        final_results['person_details'][f'person_{person_id}'] = {
            'entry_time': person_data['entry_time'],
            'last_seen': person_data['last_seen'],
            # 'service_time': person_data['service_time'],
            'anomalies': person_data['anomalies']
        }

    return final_results

@app.route('/process-video', methods=['POST'])
@cross_origin(origins='https://xp266m2r-3000.inc1.devtunnels.ms')
    
def process_video_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Save uploaded file
    video_path = os.path.join('uploads', file.filename)
    file.save(video_path)

    try:
        # Process the video
        result = process_video(video_path)
        
        # Optional: Remove the uploaded video
        os.remove(video_path)
        
        return jsonify(result), 200
    except Exception as e:
        # Log full error for debuggin
        print(f"Error processing video: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/hea", methods=['GET'])
def healthcheck():
    return jsonify({healthcheck:"True"}),200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)