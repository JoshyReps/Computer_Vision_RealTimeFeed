from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)

# 1. Load the trained .h5 classifier model
model = load_model("face_mask_classifier_model_initial.h5")

# 2. Define your class names (make sure these match the exact order from your training output)
class_names = ["mask_off", "mask_on"]

# 3. Initialize the MTCNN face detector
detector = MTCNN()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():  
    camera = cv2.VideoCapture(0)  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            # OpenCV captures in BGR, but MTCNN and your model expect RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the current frame
            faces = detector.detect_faces(rgb_frame)
            
            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()
            
            for face in faces:
                # Extract the bounding box coordinates
                x, y, width, height = face['box']
                
                # Prevent negative coordinates if a face is partially off-screen
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
                
                # Crop the face using the bounding box
                face_boundary = rgb_frame[y1:y2, x1:x2]
                
                # Skip to the next face if the crop is invalid
                if face_boundary.size == 0:
                    continue
                
                # --- PREPROCESSING THE CROPPED FACE ---
                # Resize to the 128x128 shape your CNN model expects
                resized_face = cv2.resize(face_boundary, (128, 128))
                
                # Normalize the pixel values
                normalized_face = resized_face / 255.0
                
                # Add the batch dimension: (1, 128, 128, 3)
                input_array = np.expand_dims(normalized_face, axis=0)
                
                # --- PREDICTION ---
                prediction = model.predict(input_array, verbose=0)
                predicted_name = class_names[np.argmax(prediction)]
                
                # --- VISUALIZATION ---
                # Draw the bounding box (Rectangle) around the face
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Write the predicted name above the bounding box
                cv2.putText(annotated_frame, f"{predicted_name}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Stream the final frame to the web browser
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)