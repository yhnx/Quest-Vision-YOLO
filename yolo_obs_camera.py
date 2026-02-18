"""
YOLO Object Detection on OBS Virtual Camera Feed with Speed Tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import time

class SpeedTracker:
    """
    Class to track objects and estimate their speed based on pixel movement.
    """
    def __init__(self, fps=30.0, max_distance=100, buffer_size=5):
        """
        Initialize the SpeedTracker.
        
        Args:
            fps (float): Frames per second of the video source.
            max_distance (int): Maximum distance (in pixels) to consider an object the same across frames.
            buffer_size (int): Number of missed frames allowed before removing a track.
        """
        self.fps = fps
        self.max_distance = max_distance
        self.buffer_size = buffer_size
        
        # tracks: {id: {'centroid': (x, y), 'speed': speed_val, 'bbox': (x1,y1,x2,y2), 'missed_frames': 0}}
        self.tracks = {}
        self.next_object_id = 0

    def set_fps(self, fps):
        """Update the FPS value."""
        if fps > 0:
            self.fps = fps

    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections (list): List of dicts {'centroid': (x,y), 'bbox': (x1,y1,x2,y2)}
            
        Returns:
            dict: Current active tracks.
        """
        # Mark all current detections as unmatched
        for det in detections:
            det['matched'] = False
            
        active_track_ids = []
        
        # 1. Match existing tracks to new detections
        for track_id, track_data in self.tracks.items():
            prev_cx, prev_cy = track_data['centroid']
            
            # Find closest detection
            min_dist = float('inf')
            best_match_idx = -1
            
            for idx, detection in enumerate(detections):
                if detection['matched']:
                    continue
                    
                curr_cx, curr_cy = detection['centroid']
                dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_match_idx = idx
            
            if best_match_idx != -1:
                # Match found
                match = detections[best_match_idx]
                match['matched'] = True
                
                # Calculate speed (pixels per second)
                current_speed = min_dist * self.fps
                
                # Update track
                self.tracks[track_id]['centroid'] = match['centroid']
                self.tracks[track_id]['bbox'] = match['bbox']
                self.tracks[track_id]['speed'] = current_speed 
                self.tracks[track_id]['missed_frames'] = 0
                active_track_ids.append(track_id)
            else:
                # No match found, increase missed frames
                self.tracks[track_id]['missed_frames'] += 1
        
        # 2. Create new tracks for unmatched detections
        for detection in detections:
            if not detection['matched']:
                self.tracks[self.next_object_id] = {
                    'centroid': detection['centroid'],
                    'bbox': detection['bbox'],
                    'speed': 0,
                    'missed_frames': 0
                }
                active_track_ids.append(self.next_object_id)
                self.next_object_id += 1
        
        # 3. Clean up old tracks
        clean_tracks = {}
        for track_id, track_data in self.tracks.items():
            if track_data['missed_frames'] < self.buffer_size:
                clean_tracks[track_id] = track_data
        self.tracks = clean_tracks
        
        return self.tracks

def run_tracker(camera_index=5, model_path="yolov8n.pt"):
    """
    Main function to run the YOLO tracker on a camera feed.
    """
    # Load the YOLOv8 model
    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(model_path) 
    
    # Open the Camera
    print(f"Opening camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize Tracker
    # We will update FPS once we get it from the camera or measure it
    tracker = SpeedTracker(fps=30.0)
    
    # Try to get actual FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0:
        tracker.set_fps(actual_fps)
        print(f"Camera FPS: {actual_fps}")
    else:
        print("Could not read Camera FPS, defaulting to 30.0")

    print("\nStarting Loop. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Run YOLO inference
            # verbose=False suppresses the stdout printing from YOLO
            results = model(frame, conf=0.5, verbose=False)
            
            # Extract detections for the tracker
            current_detections = []
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, coords)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                current_detections.append({'centroid': (cx, cy), 'bbox': (x1, y1, x2, y2)})
            
            # Update Tracker
            tracks = tracker.update(current_detections)

            # Visualization
            for track_id, track_data in tracks.items():
                x1, y1, x2, y2 = track_data['bbox']
                cx, cy = track_data['centroid']
                speed = track_data['speed']
                
                # Draw BBox (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Centroid
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw Speed Text
                label = f"ID: {track_id} Speed: {speed:.1f} px/s"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display FPS status
            cv2.putText(frame, f"FPS: {tracker.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLO Tracker - Speed Estimation", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Terminated.")


if __name__ == "__main__":
    # Allow command line argument for camera index
    cam_index = 0
    if len(sys.argv) > 1:
        try:
            cam_index = int(sys.argv[1])
        except ValueError:
            print(f"Value Error: Camera index must be a number")
            sys.exit(1)
    
    run_tracker(camera_index=cam_index)
