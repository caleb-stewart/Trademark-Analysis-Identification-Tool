from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(img)
    confidence_threshold = 0.25

    for box in results[0].boxes:
        conf = box.conf[0].item()
        if conf > confidence_threshold:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            color = (255, 255, 255)
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    _, img_encoded = cv2.imencode(".jpg", img)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run()