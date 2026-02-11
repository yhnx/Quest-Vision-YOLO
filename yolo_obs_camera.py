"""
YOLO Object Detection on OBS Virtual Camera Feed

"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys


def main(camera_index=5):
    """
    Args:
        camera_index: (default: 0)
        but generally obs feed at 5
    """
    
    # Load the YOLOv8 model (nano for speed, can use small, medium, large, or xlarge)
    # First run will download the model (~250MB for nano)
    model = YOLO("yolov8n.pt")  # nano model (fastest)
    # model = YOLO("yolov8s.pt")  # small model (balanced)
    # model = YOLO("yolov8m.pt")  # medium model (more accurate)
    
    # Open the OBS Virtual Camera
    print(f"Camera index: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error")
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
   
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Run YOLO inference on the frame
            results = model(frame, conf=0.5, verbose=False)
            
            # Render results on the frame (adds bounding boxes, labels, confidence scores)
            annotated_frame = results[0].plot()
            
            # Display the frame with detections
            cv2.imshow("YOLO Detection - OBS Virtual Camera", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q')
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
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Value Error: Camera index must be a number")
            sys.exit(1)
    
    main(camera_index)
