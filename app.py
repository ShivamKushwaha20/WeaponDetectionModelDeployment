from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import os
import torch
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Create upload and output folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the YOLO model
MODEL_PATH = 'modals/best.pt'  # Update this if your model path differs
model = YOLO(MODEL_PATH)
print("Model Loaded!")

# Check if file extension is allowed
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_objects(frame):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    results = model(frame)
    detections = []
    for det in results.xywh[0]:
        x, y, w, h, conf, cls = det
        label = model.names[int(cls)] if hasattr(model, 'names') else 'weapon'
        detections.append((int(x - w/2), int(y - h/2), int(w), int(h), label, conf.item()))
    return detections

# Process an image and draw rectangles
def process_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        return False, "Could not read the image"

    # Run YOLO inference and get annotated output
    results = model(input_path, save=False, verbose=False)  # Process without saving to default runs/
    if not results:
        return False, "No detections found"

    # Get the annotated image from results
    for r in results:
        annotated_img = r.plot()  # Get the image with YOLO's default annotations (boxes, labels, scores)
        # Save the annotated image
        cv2.imwrite(output_path, annotated_img)
        return True, "Image processed successfully"

    return False, "Failed to process image"

# Process a video using YOLO model
def process_video(input_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, "Could not read the video"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the frame
        results = model(frame, save=False, verbose=False)
        if results:
            for r in results:
                annotated_frame = r.plot()  # Get frame with YOLO's annotations
                out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    return True, "Video processed successfully"

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Handle image upload and processing
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))

    if file and is_allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Define output path
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Process the image
        success, message = process_image(input_path, output_path)
        if success:
            return render_template('result.html', output_file=output_filename, file_type='image')
        else:
            flash(message)
            return redirect(url_for('home'))

    flash('Invalid file format. Please upload PNG, JPG, or JPEG')
    return redirect(url_for('home'))

# Handle video upload and processing
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))

    if file and is_allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Define output path
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Process the video
        success, message = process_video(input_path, output_path)
        if success:
            return render_template('result.html', output_file=output_filename, file_type='video')
        else:
            flash(message)
            return redirect(url_for('home'))

    flash('Invalid file format. Please upload MP4')
    return redirect(url_for('home'))

# Serve the processed file for download
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)