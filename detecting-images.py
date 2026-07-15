import cv2
import os
import time
from ultralytics import YOLO

# Define paths
MODEL_PATH = r"C:\Users\JAYESH\Desktop\Weapons-and-Knives-Detector-with-YOLOv8\runs\detect\Normal_Compressed\weights\best.pt"
RESULTS_DIR = "./Results"  # Ensure results are stored properly

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def detect_objects_in_photo(image_path):
    """ Detect objects in an image and save result with a unique filename. """
    image_orig = cv2.imread(image_path)
    
    if image_orig is None:
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        print(f"Number of detections: {len(detections)}")
        
        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Generate unique filename with timestamp
    timestamp = int(time.time())
    result_path = os.path.join(RESULTS_DIR, f"detected_{timestamp}.jpg")
    
    cv2.imwrite(result_path, image_orig)
    print(f"Result saved at: {result_path}")
    return result_path

def detect_objects_in_video(video_path):
    """ Detect objects in a video and save result with a unique filename. """
    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Generate unique filename with timestamp
    timestamp = int(time.time())
    result_video_path = os.path.join(RESULTS_DIR, f"detected_video_{timestamp}.avi")

    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)

    video_capture.release()
    out.release()

    print(f"Processed video saved at: {result_video_path}")
    return result_video_path

def detect_objects_and_plot(image_path):
    """ Detect objects and display the image with bounding boxes. """
    image_orig = cv2.imread(image_path)

    if image_orig is None:
        print(f"Error: Image not found at {image_path}")
        return None

    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        print(f"Number of detections: {len(detections)}")
        
        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imshow("Detected Objects", image_orig)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
