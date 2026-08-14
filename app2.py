from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the YOLOv8 face model and Keras mask classifier
model = YOLO('face_yolo8s.pt')
loaded_model = load_model("face_mask_classifier_model.h5")
class_names = ['mask_off', 'mask_on']

@app.route('/')
def index():
    return render_template('index.html')

def classify_face(face_crop):
    # 1. Convert BGR to RGB so the model sees normal skin colors
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # 2. Resize and format for the Keras model
    resized = cv2.resize(rgb_face, (128, 128))
    image_array = np.expand_dims(resized, axis=0).astype('float32') / 255.0
    
    # 3. Predict mask on/off
    prediction = loaded_model.predict(image_array, verbose=0)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[predicted_class])
    
    return predicted_class, confidence

def gen_frames():
    camera = cv2.VideoCapture(0)  # use 0 for web camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection on the frame
            results = model(frame)

            # Visualize the results on the frame (without YOLO's default labels)
            annotated_frame = results[0].plot(labels=False, conf=False)

            # Process each detected bounding box
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls].lower()
                    
                    # Ensure we are only cropping faces (accepts 'face' or 'person' depending on your YOLO model)
                    if class_name not in ['face', 'person']:
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    
                    # Prevent out-of-bounds crops
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    try:
                        # Pass the cropped face to the mask classifier
                        predicted_class, confidence = classify_face(face_crop)
                        
                        # Create the label (e.g., "mask_on (98.5%)")
                        label = f"{class_names[predicted_class]} ({confidence * 100:.1f}%)"
                        color = colors(cls, True)
                        
                        # Draw the label on the screen
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    except Exception as e:
                        print(f"Error classifying face: {e}")
                        continue

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)