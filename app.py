from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import platform
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

os.makedirs("static/results", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ---------- Cross-platform Camera Initialization ----------
def init_camera():
    """
    Try to open a working camera on Windows / Linux / macOS.
    Returns an opened VideoCapture object or None.
    """
    system = platform.system()

    # Preferred backends per OS
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Darwin":          # macOS
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:                             # Linux and others
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # Try indices 0 → 4 with each backend
    for backend in backends:
        for index in range(5):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Verify we can actually read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                # Optional: set a reasonable resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"✓ Camera opened: index={index}, backend={backend}")
                return cap

            cap.release()

    print("⚠ No usable camera found")
    return None


camera = init_camera()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_frames():
    """MJPEG stream generator with black-frame fallback."""
    if camera is None:
        # Continuous black frame so the browser never hangs
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Available", (400, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        return

    while True:
        success, frame = camera.read()
        if not success:
            # Camera disconnected mid-stream → black frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Lost", (480, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            frame = cv2.resize(frame, (1280, 720))

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def analyze_image(img):
    img = cv2.resize(img, (800, 600))
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Make sure rice grains are white
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

        # Simple HSV rule for rice (yellowish / light)
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

    cv2.putText(original, f"Purity: {purity}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(original, f"Rice: {rice_count}  Foreign: {foreign_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    if not success or frame is None:
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
    try:
        app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
    finally:
        if camera is not None:
            camera.release()
            print("Camera released")