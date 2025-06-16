import cv2
import numpy as np
from IPython.display import display, HTML, clear_output
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import time
import mediapipe as mp

# Install MediaPipe if needed
try:
    import mediapipe
except ImportError:
    !pip install mediapipe
    import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define facial expression analysis function
def analyze_expression(face_landmarks, frame_shape):
    """Analyze facial expression based on landmark positions"""
    # Get key landmarks
    landmarks = face_landmarks.landmark
    
    # Calculate mouth openness
    mouth_upper = landmarks[13]
    mouth_lower = landmarks[14]
    mouth_openness = abs(mouth_upper.y - mouth_lower.y) * frame_shape[0]
    
    # Calculate eyebrow raise
    left_eyebrow = landmarks[105]
    right_eyebrow = landmarks[334]
    eyebrow_raise = (left_eyebrow.y + right_eyebrow.y) * frame_shape[0] / 2
    
    # Calculate smile intensity
    left_mouth_corner = landmarks[61]
    right_mouth_corner = landmarks[291]
    mouth_corners_distance = abs(left_mouth_corner.x - right_mouth_corner.x) * frame_shape[1]
    
    # Determine expression
    expression = "Neutral"
    
    if mouth_openness > 15:
        expression = "Surprised" if eyebrow_raise < 100 else "Yawning"
    elif mouth_corners_distance > 50:
        expression = "Smiling"
    elif eyebrow_raise < 100:
        expression = "Frowning"
    
    # Calculate expression intensity
    intensity = min(1.0, max(0.0, 
        (mouth_openness/20 + (150 - eyebrow_raise)/50 + mouth_corners_distance/60) / 3
    ))
    
    return expression, intensity

# Setup webcam with frame capture function
display(HTML('''
<video id="video" width="640" height="480" autoplay style="display:none;"></video>
<canvas id="canvas" style="display:none;"></canvas>
<button id="stopButton" style="padding:10px;font-size:16px;margin:10px;background:#4CAF50;color:white;border:none;border-radius:5px;">
    Stop Monitoring
</button>
<div id="status" style="padding:10px;font-family:Arial;">Initializing camera...</div>
<script>
// Create monitoring state on window object
window.monitoringActive = true;
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
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
        statusDiv.textContent = 'Camera ready. Starting monitoring...';
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
});

// Initialize camera
setupCamera();
</script>
'''))

# Define state colors with requested changes:
# - Looking away (left/right) is now red
# - Looking down is now green (engaged)
state_colors = {
    'looking_forward': (0, 255, 0),      # Green - Engaged
    'looking_away': (0, 0, 255),          # Red - Disengaged (looking left/right)
    'face_not_visible': (128, 128, 128)   # Gray - Unknown
}

print("Starting student engagement monitoring...")
print("Click 'Stop' button to exit")

def get_frame():
    """Capture frame using predefined JavaScript function"""
    return eval_js("captureFrame()")

def detect_faces(frame):
    """Detect faces using Haar cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

def detect_eyes(face_roi):
    """Detect eyes within face region"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

def determine_state(face_position, eyes, frame_width):
    """Determine student's attention state with requested changes"""
    x, y, w, h = face_position
    
    # No eyes detected = looking down, now considered engaged (green)
    if len(eyes) == 0:
        return 'looking_forward'
    
    # Calculate horizontal position
    face_center_x = x + w//2
    frame_center_x = frame_width // 2
    position_ratio = (face_center_x - frame_center_x) / frame_center_x
    
    # Determine direction based on position
    if position_ratio < -0.3 or position_ratio > 0.3: 
        return 'looking_away'  # Looking left/right is now disengaged (red)
    
    return 'looking_forward'  # Centered is engaged (green)

# Initialize tracking
state_counts = {state: 0 for state in state_colors}
expression_counts = {
    'Neutral': 0,
    'Smiling': 0,
    'Surprised': 0,
    'Frowning': 0,
    'Yawning': 0,
    'Other': 0
}
engagement_weights = {
    'looking_forward': 1.0,     # Engaged (green)
    'looking_away': 0.4,        # Disengaged (red)
    'face_not_visible': 0.5     # Unknown (gray)
}
frame_count = 0
engagement_scores = []

try:
    print("Initializing camera...")
    time.sleep(3)  # Allow camera initialization
    
    while frame_count < 300:  # Process up to 300 frames
        frame_count += 1
        
        # Check if monitoring should stop using JS function
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
        frame_width = frame.shape[1]
        height, width = frame.shape[:2]
        
        # Detect faces and eyes
        faces = detect_faces(frame)
        current_states = []
        current_expression = "None"
        expression_intensity = 0
        
        # Process each detected face
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                    
                eyes = detect_eyes(face_roi)
                state = determine_state((x, y, w, h), eyes, frame_width)
                current_states.append(state)
                state_counts[state] += 1
                
                # Analyze facial expression using MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Analyze expression
                        expression, intensity = analyze_expression(face_landmarks, frame.shape)
                        current_expression = expression
                        expression_intensity = intensity
                        expression_counts[expression] += 1
                        
                        # Draw facial landmarks
                        for landmark in face_landmarks.landmark:
                            cx, cy = int(landmark.x * width), int(landmark.y * height)
                            cv2.circle(annotated_frame, (cx, cy), 1, (0, 255, 255), -1)
                
                # Annotate face
                color = state_colors[state]
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                
                # Use appropriate label based on state
                if state == 'looking_away':
                    label = "Looking Away"
                elif state == 'looking_forward':
                    # Distinguish between engaged and looking down
                    label = "Engaged" if len(eyes) > 0 else "Looking Down (Engaged)"
                else:
                    label = state.replace('_', ' ').title()
                    
                # Add expression label if detected
                text_y = y - 10
                cv2.putText(annotated_frame, label, 
                            (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if current_expression != "None":
                    text_y -= 30
                    expression_text = f"Expression: {current_expression} ({expression_intensity:.1f})"
                    cv2.putText(annotated_frame, expression_text, 
                                (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                
                # Annotate eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(annotated_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
        else:
            state = 'face_not_visible'
            current_states.append(state)
            state_counts[state] += 1
        
        # Calculate engagement with expression bonus
        if current_states:
            base_score = np.mean([engagement_weights[s] for s in current_states])
            
            # Add bonus for positive expressions
            expression_bonus = 0
            if current_expression == "Smiling":
                expression_bonus = 0.2 * expression_intensity
            elif current_expression == "Surprised":
                expression_bonus = 0.1 * expression_intensity
            
            avg_score = min(1.0, base_score + expression_bonus)
            engagement_scores.append(avg_score)
        else:
            avg_score = 0.0
        
        # Add metrics to frame
        cv2.putText(annotated_frame, f"Engagement: {avg_score:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/300", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'Stop' to end monitoring", 
                   (width - 300, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display results
        clear_output(wait=True)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        display(HTML(f'<img src="data:image/jpeg;base64,{b64encode(buffer).decode()}" />'))
        
        # Print frame summary
        summary = f"Frame {frame_count}: Engagement: {avg_score:.2f}"
        if current_expression != "None":
            summary += f", Expression: {current_expression} ({expression_intensity:.1f})"
        print(summary)
        
        time.sleep(0.1)

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())

# Final report
if engagement_scores:
    avg_engagement = np.mean(engagement_scores)
    print("\n" + "="*50)
    print("Engagement Report Summary")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Average engagement: {avg_engagement:.2f}/1.0")
    
    print("\nAttention States:")
    for state, count in state_counts.items():
        percentage = count / frame_count * 100
        state_label = {
            'looking_forward': 'Engaged (Looking Forward/Down)',
            'looking_away': 'Disengaged (Looking Away)',
            'face_not_visible': 'Face Not Visible'
        }[state]
        print(f"- {state_label}: {count} frames ({percentage:.1f}%)")
    
    print("\nFacial Expression Distribution:")
    total_expression_frames = sum(expression_counts.values())
    if total_expression_frames > 0:
        for expression, count in expression_counts.items():
            percentage = count / total_expression_frames * 100
            print(f"- {expression}: {count} frames ({percentage:.1f}%)")
    else:
        print("- No expressions detected in any frames")
    
    # Calculate expression diversity
    detected_expressions = sum(1 for count in expression_counts.values() if count > 0)
    diversity_score = detected_expressions / len(expression_counts) * 100
    print(f"\nExpression Diversity: {diversity_score:.1f}%")
    
    print("="*50)
else:
    print("No frames processed. Check camera connection.")
