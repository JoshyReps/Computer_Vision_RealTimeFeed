# Real-time Object Detection

A real-time object detection web app built with Flask, OpenCV, and the Ultralytics YOLOv8 model. The app captures frames from a camera, detects objects in each frame, and streams the annotated video feed to a browser.

## Requirements

> **A camera is required.** This application detects objects from a live video feed, so you must have a camera available — either a built-in webcam, an external USB camera, or an integrated camera on a laptop. The app uses the default camera device (`0`), so make sure your camera is connected and working before starting.

Other requirements:

- Python 3.8+
- A working camera
- [PyTorch](https://pytorch.org/) (installed via `requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/eemberda/cv_realtime_object_detection.git
   cd cv_realtime_object_detection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the YOLOv8 weights file (`yolov8n.pt`) is in the project root. If it is missing, download it:

   ```bash
   curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -o yolov8n.pt
   ```

## Usage

1. Make sure your camera is connected and accessible.
2. Run the app:

   ```bash
   python app.py
   ```

3. Open your browser and go to <http://127.0.0.1:5000>.

You should see a live video feed with bounding boxes and class labels drawn around detected objects.

## How it works

- `app.py` loads the YOLOv8 model and reads frames from the camera (`cv2.VideoCapture(0)`).
- Each frame is passed through the model, and the detections are annotated directly onto the frame.
- The annotated frames are streamed to the browser as a MJPEG feed (`/video_feed`), which the frontend displays via an `<img>` tag in `templates/index.html`.

## Troubleshooting

- **No video / black screen:** your camera may be in use by another application, or no camera is detected. Close other apps using the camera and verify it works with your system camera app first.
- **"Cannot identify the input file" errors:** your camera may not be connected. Check the camera index in `app.py` (line 15) — change `0` if your camera is on a different device index.
- **Slow performance:** the model is loaded on CPU by default. For faster inference, install a CUDA-enabled PyTorch build and use a compatible GPU.

## License

MIT
