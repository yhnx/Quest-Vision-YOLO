"""
Helper script to find available camera devices and view their feeds.
Useful for identifying which camera index corresponds to the OBS Virtual Camera.
"""

import cv2
import sys


def find_available_cameras(max_cameras=10):
    """
    Scan for available camera devices.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    print("Scanning for available cameras...")
    print("-" * 50)
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # G
            # et some properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Camera {i}: {width}x{height} @ {fps} FPS")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")
    
    print("-" * 50)
    
    if available_cameras:
        print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
        print(f"\nTo view a camera feed, run:")
        print(f"  python find_cameras.py <camera_index>")
        print(f"\nExample:")
        print(f"  python find_cameras.py {available_cameras[-1]}")
    else:
        print("\nNo cameras found! Make sure OBS Virtual Camera is installed and enabled.")
    
    return available_cameras


def view_camera_feed(camera_index):
    """
    Open and display a live camera feed.
    
    Args:
        camera_index: Index of the camera to open
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        print("\nFirst, run without arguments to find available cameras:")
        print("  python find_cameras.py")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nOpened Camera {camera_index}: {width}x{height} @ {fps} FPS")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("-" * 50)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(
                frame,
                f"Camera {camera_index} | Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display the frame
            cv2.imshow(f"Camera Feed - Index {camera_index}", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('s'):
                filename = f"camera_{camera_index}_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera {camera_index} closed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # View specific camera feed
        camera_index = int(sys.argv[1])
        view_camera_feed(camera_index)
    else:
        # Just scan for cameras
        find_available_cameras()
