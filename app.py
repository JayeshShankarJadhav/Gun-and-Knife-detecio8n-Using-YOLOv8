import os
import cv2
from flask import Flask, render_template, request, send_from_directory, Response
from ultralytics import YOLO

app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("runs/detect/Normal_Compressed/weights/best.pt")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    file_ext = filename.split('.')[-1].lower()

    if file_ext in ['jpg', 'jpeg', 'png']:
        results = model(filepath)
        result_path = os.path.join(RESULT_FOLDER, filename)
        for result in results:
            im_array = result.plot()
            cv2.imwrite(result_path, im_array)
        return render_template('result.html', detected_file=filename, file_type="image")

    elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
        result_video_path = os.path.join(RESULT_FOLDER, filename)
        process_video(filepath, result_video_path)
        return render_template('result.html', detected_file=filename, file_type="video")

    else:
        return "Unsupported file format", 400


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            detected_frame = result.plot()

        if 'detected_frame' in locals():
            out.write(detected_frame)

    cap.release()
    out.release()


@app.route('/results/<filename>')
def show_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/live')
def live():
    return render_template('live.html')


def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        for result in results:
            frame = result.plot()  # Overlay detections

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
