# Install required packages
!pip install openvino-dev[onnx]
!pip install mediapipe

import cv2
import numpy as np
import time
from openvino.runtime import Core
from IPython.display import display, HTML
from base64 import b64decode, b64encode
import os

# Download OpenVINO models
if not os.path.exists('model'):
    os.makedirs('model')

!omz_downloader --name face-detection-adas-0001 --precisions FP32 -o model
!omz_downloader --name facial-landmarks-35-adas-0002 --precisions FP32 -o model
!omz_downloader --name emotions-recognition-retail-0003 --precisions FP32 -o model

# Initialize OpenVINO core
ie = Core()

# Load models
print("Loading OpenVINO models...")
face_model = ie.read_model(model='model/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml')
face_compiled_model = ie.compile_model(model=face_model, device_name="CPU")
face_input_layer = face_compiled_model.input(0)
_, _, H, W = face_input_layer.shape
print(f"Face detection input shape: {H}x{W}")

landmark_model = ie.read_model(model='model/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml')
landmark_compiled_model = ie.compile_model(model=landmark_model, device_name="CPU")
landmark_input_layer = landmark_compiled_model.input(0)
_, _, LANDMARK_H, LANDMARK_W = landmark_input_layer.shape
print(f"Landmark model input shape: {LANDMARK_H}x{LANDMARK_W}")

emotion_model = ie.read_model(model='model/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml')
emotion_compiled_model = ie.compile_model(model=emotion_model, device_name="CPU")
emotion_input_layer = emotion_compiled_model.input(0)
_, _, EMOTION_H, EMOTION_W = emotion_input_layer.shape
print(f"Emotion model input shape: {EMOTION_H}x{EMOTION_W}")

emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']

print("Models loaded successfully!")

# Setup webcam with JavaScript
display(HTML('''
<div style="display: flex; flex-direction: column; align-items: center;">
    <div>
        <video id="video" width="640" height="480" autoplay style="display:none;"></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <img id="outputImage" width="640" height="480" style="border: 2px solid #4CAF50; border-radius: 5px;">
    </div>
    <div style="margin-top: 20px;">
        <button id="stopButton" style="padding:10px;font-size:16px;margin:10px;background:#f44336;color:white;border:none;border-radius:5px;">
            Stop Monitoring
        </button>
        <div id="status" style="padding:10px;font-family:Arial;font-size:16px;">Initializing camera...</div>
    </div>
</div>
<script>
// Create monitoring state on window object
window.monitoringActive = true;
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const outputImage = document.getElementById('outputImage');
const stopButton = document.getElementById('stopButton');
const statusDiv = document.getElementById('status');

// Function to capture frame
function captureFrame() {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        return '';
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Function to update output image
function updateOutputImage(dataUrl) {
    outputImage.src = dataUrl;
}

// Function to check monitoring state
function isMonitoringActive() {
    return window.monitoringActive;
}

// Camera setup
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        video.srcObject = stream;
        statusDiv.textContent = 'Camera ready. Monitoring active...';
        return true;
    } catch (err) {
        statusDiv.textContent = 'Camera error: ' + err.message;
        return false;
    }
}

// Stop button handler
stopButton.addEventListener('click', function() {
    window.monitoringActive = false;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    statusDiv.textContent = 'Monitoring stopped.';
    stopButton.textContent = 'Stopped';
    stopButton.disabled = true;
    stopButton.style.background = '#9e9e9e';
});

// Initialize camera
setupCamera();
</script>
'''))

# Define state colors
state_colors = {
    'engaged': (0, 255, 0),          # Green - Engaged
    'disengaged': (0, 0, 255),        # Red - Disengaged
    'looking_down': (255, 255, 0),    # Yellow - Looking down
    'unknown': (128, 128, 128)        # Gray - Unknown
}

print("Starting student engagement monitoring with OpenVINO...")
print("Click 'Stop' button to exit")

def get_frame():
    """Capture frame using predefined JavaScript function"""
    return eval_js("captureFrame()")

def update_output(data_url):
    """Update output image in browser"""
    display(HTML(f"<script>updateOutputImage('{data_url}');</script>"))

# ----------------------------
# Face Detection with OpenVINO
# ----------------------------
def detect_faces_ov(frame):
    # Preprocess input image
    resized = cv2.resize(frame, (W, H))
    input_image = np.expand_dims(resized.transpose(2, 0, 1), axis=0)
    
    # Inference
    results = face_compiled_model([input_image])[face_compiled_model.output(0)]
    faces = []
    
    # Process results
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.3:  # Lower confidence threshold
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            w = x_max - x_min
            h = y_max - y_min
            # Filter out small detections
            if w > 30 and h > 30:  # More lenient size filter
                faces.append((x_min, y_min, w, h))
    
    return faces

# ----------------------------
# Facial Landmarks with OpenVINO
# ----------------------------
def detect_landmarks_ov(face_roi):
    # Preprocess input image with CORRECT dimensions
    resized = cv2.resize(face_roi, (LANDMARK_W, LANDMARK_H))
    input_image = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    # Inference
    landmarks = landmark_compiled_model([input_image])[landmark_compiled_model.output(0)]
    return landmarks[0].reshape(-1, 2)

# ----------------------------
# Emotion Recognition with OpenVINO
# ----------------------------
def detect_emotion_ov(face_roi):
    # Preprocess input image with CORRECT dimensions
    resized = cv2.resize(face_roi, (EMOTION_W, EMOTION_H))
    input_image = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
    
    # Inference
    emotions = emotion_compiled_model([input_image])[emotion_compiled_model.output(0)]
    emotion_idx = np.argmax(emotions)
    confidence = emotions[0][emotion_idx]
    
    return emotion_labels[emotion_idx], confidence

# ----------------------------
# Attention State Detection (Simplified)
# ----------------------------
def determine_state(face_position, frame_width):
    x, y, w, h = face_position
    
    # Calculate face center
    face_center_x = x + w//2
    frame_center_x = frame_width // 2
    position_ratio = (face_center_x - frame_center_x) / frame_center_x

    # Simple attention detection
    if abs(position_ratio) > 0.3:
        return 'disengaged'
    
    return 'engaged'

# Initialize tracking
frame_count = 0
start_time = time.time()
student_counts = []
emotion_counts = {e: 0 for e in emotion_labels}
state_counts = {s: 0 for s in state_colors}

try:
    print("Initializing camera...")
    time.sleep(3)  # Allow camera initialization

    while frame_count < 300:  # Process up to 300 frames
        frame_count += 1

        # Check if monitoring should stop
        if not eval_js('isMonitoringActive()'):
            print("Monitoring stopped by user")
            break

        # Get frame from JS
        js_reply = get_frame()
        if not js_reply or ',' not in js_reply:
            continue

        # Decode image
        _, data = js_reply.split(',', 1)
        nparr = np.frombuffer(b64decode(data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            continue

        # Mirror and process frame
        frame = cv2.flip(frame, 1)
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Detect faces with OpenVINO
        faces = detect_faces_ov(frame)
        student_count = len(faces)
        student_counts.append(student_count)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Get face ROI
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            try:
                # Detect emotion
                emotion, confidence = detect_emotion_ov(face_roi)
                emotion_counts[emotion] += 1
                
                # Determine attention state
                state = determine_state((x, y, w, h), width)
                state_counts[state] = state_counts.get(state, 0) + 1
                
                # Draw face bounding box
                color = state_colors.get(state, (128, 128, 128))
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 3)  # Thicker border
                
                # Add labels
                label = f"{state.capitalize()} | {emotion.capitalize()}"
                text_y = y - 10 if y > 30 else y + 30
                cv2.putText(annotated_frame, label, (x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Larger font
                
                # Add confidence indicator
                cv2.putText(annotated_frame, f"{confidence*100:.1f}%", 
                           (x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
                
            except Exception as e:
                # Skip this face but continue processing
                continue
        
        # Add metrics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Students: {student_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/300", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add status indicator
        status_text = "Monitoring Active" if eval_js('isMonitoringActive()') else "Monitoring Stopped"
        status_color = (0, 255, 0) if eval_js('isMonitoringActive()') else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (width - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display results
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        data_url = f"data:image/jpeg;base64,{b64encode(buffer).decode()}"
        update_output(data_url)

        # Print periodic summary
        if frame_count % 10 == 0:  # Print every 10 frames
            print(f"Frame {frame_count}: Faces: {student_count}")

        time.sleep(0.01)

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())

# Final report
if frame_count > 0:
    avg_students = np.mean(student_counts) if student_counts else 0
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("CLASSROOM MONITORING REPORT")
    print("="*50)
    print(f"Monitoring duration: {elapsed_time:.1f} seconds")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed_time:.1f if elapsed_time > 0 else 0}")
    print(f"Average students detected: {avg_students:.1f}")

    print("\nATTENTION STATES DISTRIBUTION:")
    total_states = sum(state_counts.values())
    for state, count in state_counts.items():
        percentage = (count / total_states) * 100 if total_states > 0 else 0
        print(f"- {state.upper()}: {count} frames ({percentage:.1f}%)")

    print("\nEMOTION DISTRIBUTION:")
    total_emotions = sum(emotion_counts.values())
    for emotion, count in emotion_counts.items():
        percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
        print(f"- {emotion.upper()}: {count} frames ({percentage:.1f}%)")

    print("\n" + "="*50)
    print("MONITORING COMPLETE")
    print("="*50)
else:
    print("No frames processed. Please check camera access and try again.")
