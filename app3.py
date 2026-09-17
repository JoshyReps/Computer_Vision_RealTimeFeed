from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# 1. Setup 10 Classes and Load Keras Model
class_names = [
    "Alaxan", 
    "Bactidol", 
    "Bioflu", 
    "Biogesic", 
    "DayZinc", 
    "Decolgen", 
    "Fish Oil", 
    "Kremil S", 
    "Neozep", 
    "ShortTest"
]

# We will load the model inside the route or wait until it exists
try:
    model = load_model('pill_classifier_model.h5')
    print("Successfully loaded pill_classifier_model.h5")
except Exception as e:
    print("Warning: Could not load pill_classifier_model.h5 yet. Make sure it's in the folder!")
    model = None

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
            h, w, _ = frame.shape
            
            # --- TARGET BOX ---
            box_size = 250
            x1 = int(w / 2 - box_size / 2)
            y1 = int(h / 2 - box_size / 2)
            x2 = int(w / 2 + box_size / 2)
            y2 = int(h / 2 + box_size / 2)
            
            # Crop target area
            roi = frame[y1:y2, x1:x2]
            
            if model is None:
                color = (0, 0, 255)
                label_text = "Waiting for model..."
            else:
                # --- PREPROCESSING FOR KERAS ---
                # Convert to RGB, resize to 128x128, normalize to [0,1]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(roi_rgb, (128, 128))
                input_array = np.expand_dims(resized_img, axis=0) / 255.0
                
                # --- PREDICTION ---
                prediction = model.predict(input_array, verbose=0)[0]
                predicted_idx = int(np.argmax(prediction))
                conf_score = float(prediction[predicted_idx]) * 100
                predicted_name = class_names[predicted_idx]
                
                # --- VISUALIZATION ---
                if conf_score > 99.5:
                    color = (0, 255, 0)
                    label_text = f"{predicted_name} ({conf_score:.1f}%)"
                else:
                    color = (0, 0, 255)
                    label_text = "Scanning..."
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)