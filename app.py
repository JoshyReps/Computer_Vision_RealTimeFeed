from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():  
    camera = cv2.VideoCapture(0)  # use 0 for web camera
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Perform object detection on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True) 