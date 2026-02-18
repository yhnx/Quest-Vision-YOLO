# YOLO Object Detection on OBS Virtual Camera

This project runs YOLOv8 object detection on an OBS (Open Broadcaster Software) virtual camera feed in real-time.

## Prerequisites

### 1. Install OBS Virtual Camera

- Download and install [OBS Studio](https://obsproject.com/)
- Install the OBS Virtual Camera plugin:
  - On Windows: Use the built-in virtual camera feature (OBS 28+) or install the community plugin
  - Enable it in OBS: Tools → Start Virtual Camera

### 2. Install Python Dependencies

```bash
# Option 1: Using requirements.txt
pip install -r requirements.txt

# Option 2: Manual installation
pip install ultralytics opencv-python numpy torch torchvision
```

## Usage

### Step 1: Find Your Camera Index (Recommended)

First, identify which camera index corresponds to your OBS Virtual Camera:

```bash
python find_cameras.py
```

This will scan and list all available cameras with their resolutions and FPS.

### Step 2: Run YOLO Detection

```bash
# Using default camera (usually index 0)
python yolo_obs_camera.py

# Using a specific camera index
python yolo_obs_camera.py 1

# Example: If find_cameras.py shows OBS Virtual Camera at index 2
python yolo_obs_camera.py 2
```

### Step 3: Interact with the Program

While the program is running:

- **Press 'q'**: Quit the program

- The program displays real-time detection results with bounding boxes

## How It Works
    
### Script Overview

The `yolo_obs_camera.py` script performs these operations:

1. **Load YOLOv8 Model**: Initializes a pre-trained YOLOv8 model (default `yolov8n.pt`).
2. **Open Camera**: Connects to the OBS virtual camera using OpenCV.
3. **Object Tracking**: Uses a `SpeedTracker` class to track objects across frames based on centroid distance.
4. **Speed Estimation**: Calculates the pixel speed of tracked objects.
5. **Visualization**: Draws bounding boxes and speed labels on the frame (classification labels are hidden).

### Modular Design

The code is structured for modularity:

- **`SpeedTracker` Class**: A reusable class that handles object tracking and speed calculation. Can be easily imported into other projects.
- **`run_tracker` Function**: The main driver function that handles camera setup and the processing loop.

### Key Parameters

In `yolo_obs_camera.py`:

```python
# Initialize Tracker with specific FPS
tracker = SpeedTracker(fps=30.0, max_distance=100)
```

- **`fps`**: The frames per second used for speed calculations. The script attempts to read this from the camera, but defaults to 30.0 if unavailable.
- **`max_distance`**: Maximum pixel distance to consider an object the same between frames.

## Troubleshooting

### Camera Not Found

- Run `python find_cameras.py` to identify available cameras
- Make sure OBS Virtual Camera is enabled in OBS
- Try different camera indices: `python yolo_obs_camera.py 0`, `python yolo_obs_camera.py 1`, etc.

### Poor Performance / Low FPS

- Use a smaller model: Change to `yolov8n.pt` (nano is fastest)
- Reduce frame resolution: Edit `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` in the script
- Increase confidence threshold to skip borderline detections

### YOLO Model Download Issues

- The first run will download the model (~250MB for nano)
- Ensure stable internet connection
- Models are cached in `~/.yolo/` directory

## Model Performance Comparison

| Model            | Speed     | Accuracy  | Size   |
| ---------------- | --------- | --------- | ------ |
| YOLOv8n (nano)   | Very Fast | Good      | ~250MB |
| YOLOv8s (small)  | Fast      | Better    | ~360MB |
| YOLOv8m (medium) | Moderate  | Very Good | ~610MB |
| YOLOv8l (large)  | Slow      | Excellent | ~940MB |

## Files

- `yolo_obs_camera.py`: Main script for real-time detection and tracking
- `find_cameras.py`: Helper script to identify available cameras
- `requirements.txt`: Python package dependencies
- `README.md`: This file

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [OBS Virtual Camera Documentation](https://obsproject.com/)
