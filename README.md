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
- **Press 's'**: Save the current frame with detections to a JPEG file
- The program displays real-time detection results with bounding boxes

## How It Works

### Script Overview

The `yolo_obs_camera.py` script performs these operations:

1. **Load YOLOv8 Model**: Initializes a pre-trained YOLOv8 model
   - `yolov8n.pt` (nano): Fastest, ~250MB, good for real-time
   - `yolov8s.pt` (small): Balanced speed/accuracy
   - `yolov8m.pt` (medium): Better accuracy, slower
   - `yolov8l.pt` (large): High accuracy, slower

2. **Open Camera**: Connects to the OBS virtual camera using OpenCV

3. **Main Loop**: For each video frame:
   - Captures frame from camera
   - Runs YOLO inference (object detection)
   - Draws bounding boxes and labels on the frame
   - Displays the annotated frame
   - Listens for keyboard commands

4. **Cleanup**: Properly releases camera resources when done

### Key Parameters

In `yolo_obs_camera.py`:

```python
model = YOLO("yolov8n.pt")  # Change model size here
results = model(frame, conf=0.5, verbose=False)  # conf=0.5 is confidence threshold
```

- **conf=0.5**: Only show detections with >50% confidence (adjust 0.0-1.0)
- Lower confidence = more detections (including false positives)
- Higher confidence = fewer detections (only high-confidence ones)

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

## Advanced Usage

### Custom Detection Classes

Modify the script to only detect specific object classes:

```python
# In main loop, after getting results:
for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        if cls in [0, 1, 2]:  # Only person, bicycle, car
            # Process this detection
```

### Change Confidence Threshold

```python
results = model(frame, conf=0.3)  # Lower = more detections
results = model(frame, conf=0.7)  # Higher = fewer, more confident detections
```

### Record Output Video

See `yolo_obs_camera_record.py` for video recording capability.

## Files

- `yolo_obs_camera.py`: Main script for real-time detection
- `find_cameras.py`: Helper script to identify available cameras
- `requirements.txt`: Python package dependencies
- `README.md`: This file

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [OBS Virtual Camera Documentation](https://obsproject.com/)
