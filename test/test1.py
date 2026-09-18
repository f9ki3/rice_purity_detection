from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

os.makedirs("static/results", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ---------- Camera ----------
camera = None
try:
    camera = cv2.VideoCapture("/dev/video0")
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = None
        print("⚠ No camera found")
except Exception as e:
    camera = None
    print(f"Camera error: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    if camera is None:
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.resize(frame, (1280, 720))
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

def analyze_image(img):
    img = cv2.resize(img, (800, 600))
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rice_count = 0
    foreign_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)

        x, y, w, h = cv2.boundingRect(cnt)

        if 15 <= mean_h <= 45 and mean_s > 35 and mean_v > 70:
            label = "Rice"
            color = (0, 255, 0)
            rice_count += 1
        else:
            label = "Foreign"
            color = (0, 0, 255)
            foreign_count += 1

        cv2.drawContours(original, [cnt], -1, color, 2)
        cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)
        cv2.putText(original, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total = rice_count + foreign_count
    purity = round((rice_count / total * 100), 1) if total > 0 else 0.0
    contamination = round(100 - purity, 1) if total > 0 else 0.0

    # Overlay 
    cv2.putText(original, f"Purity: {purity}%", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return original, thresh, rice_count, foreign_count, total, purity, contamination

@app.route("/")
def index():
    return render_template("index.html", has_camera=camera is not None)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    if camera is None:
        return "No camera available", 400

    success, frame = camera.read()
    if not success:
        return "Failed to capture frame", 500

    labeled, thresh, rice, foreign, total, purity, contamination = analyze_image(frame)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = f"static/results/original_{timestamp}.jpg"
    labeled_path = f"static/results/labeled_{timestamp}.jpg"
    thresh_path = f"static/results/thresh_{timestamp}.jpg"

    cv2.imwrite(original_path, cv2.resize(frame, (800, 600)))
    cv2.imwrite(labeled_path, labeled)
    cv2.imwrite(thresh_path, thresh)

    return render_template(
        "results.html",
        original=original_path,
        labeled=labeled_path,
        thresh=thresh_path,
        rice=rice,
        foreign=foreign,
        total=total,
        purity=purity,
        contamination=contamination,
        source="Camera Capture"
    )

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            return "Could not read the uploaded image", 400

        labeled, thresh, rice, foreign, total, purity, contamination = analyze_image(img)

        original_path = f"static/results/original_{timestamp}.jpg"
        labeled_path = f"static/results/labeled_{timestamp}.jpg"
        thresh_path = f"static/results/thresh_{timestamp}.jpg"

        cv2.imwrite(original_path, cv2.resize(img, (800, 600)))
        cv2.imwrite(labeled_path, labeled)
        cv2.imwrite(thresh_path, thresh)

        return render_template(
            "results.html",
            original=original_path,
            labeled=labeled_path,
            thresh=thresh_path,
            rice=rice,
            foreign=foreign,
            total=total,
            purity=purity,
            contamination=contamination,
            source="Uploaded Image"
        )

    return "Invalid file type. Allowed: png, jpg, jpeg, bmp, webp", 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)