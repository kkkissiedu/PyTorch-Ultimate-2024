from ultralytics import YOLO
import torch

def test_yolo_environment():
    """
    Tests the virtual environment by running inference with a YOLOv10 model.
    """
    # Check if a CUDA-enabled GPU is available and print its name
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        print("Running inference on GPU.")
    else:
        print("No GPU found. Running inference on CPU.")

    # Load the YOLOv10n (nano) model. 
    # It will be downloaded automatically on the first run.
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection for the first run.")
        return

    # Define a source image from the web for testing
    source_image = 'https://ultralytics.com/images/bus.jpg'

    # Run inference on the image
    print(f"\nPerforming object detection on: {source_image}")
    results = model(source_image)

    # Show the results in a new window
    # The window will display the image with bounding boxes and labels.
    results[0].show()

    print("\nDetection complete. A window with the results should have appeared.")
    print("Result images are also saved automatically in a 'runs/detect/' directory.")
    print("\nYour environment is set up correctly for modern YOLO models!")


if __name__ == '__main__':
    test_yolo_environment()