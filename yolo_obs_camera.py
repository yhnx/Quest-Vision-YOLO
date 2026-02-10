"""
YOLO Object Detection on OBS Virtual Camera Feed

"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys


def main(camera_index=0):
    """
    Args:
        camera_index: (default: 0)
    """
    
    # Load the YOLOv8 model (nano for speed, can use small, medium, large, or xlarge)
    # First run will download the model (~250MB for nano)
    model = YOLO("yolov8n.pt")  # nano model (fastest)
    # model = YOLO("yolov8s.pt")  # small model (balanced)
    # model = YOLO("yolov8m.pt")  # medium model (more accurate)
    
    # Open the OBS Virtual Camera
    # Usually camera index 0 is the default camera, but OBS virtual camera might be 1, 2, etc.
    # Try different indices if 0 doesn't work
    print(f"Camera index: {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error")
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
   
    print("Press 'q' to quit")
    
    frame_count = 0
    
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
            
            frame_count += 1
            
            # Add frame counter and FPS information
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display the frame with detections
            cv2.imshow("YOLO Detection - OBS Virtual Camera", annotated_frame)
            
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
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Value Error: Camera index must be a number, got '{sys.argv[1]}'")
            sys.exit(1)
    
    main(camera_index)
