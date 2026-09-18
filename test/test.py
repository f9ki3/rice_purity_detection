import cv2
import numpy as np

# =====================================
# LOAD IMAGE
# =====================================
image_path = "./test/test6.jpg"          # change to your image name

img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# Resize for consistent processing (optional)
img = cv2.resize(img, (800, 600))
original = img.copy()

# =====================================
# CONVERT TO HSV
# =====================================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# =====================================
# SEGMENT OBJECTS
# =====================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Invert if background is brighter
if np.mean(thresh) > 127:
    thresh = cv2.bitwise_not(thresh)

# Morphological cleaning
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# =====================================
# FIND CONTOURS
# =====================================
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

rice_count = 0
foreign_count = 0

# =====================================
# ANALYZE EACH OBJECT
# =====================================
for cnt in contours:
    area = cv2.contourArea(cnt)

    # Ignore very small noise
    if area < 80:
        continue

    # Mask for current object
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Mean HSV of the object
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)

    # ---------------------------------
    # COLOR CLASSIFICATION
    # ---------------------------------

    # Yellow / Puddy rice
    if 15 <= mean_h <= 45 and mean_s > 35 and mean_v > 70:
        label = "Rice"
        color = (0, 255, 0)          # Green
        rice_count += 1
    else:
        # Everything else = Foreign object
        label = "Foreign"
        color = (0, 0, 255)          # Red
        foreign_count += 1

    # Draw results
    cv2.drawContours(original, [cnt], -1, color, 2)
    cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        original,
        label,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    # Debug print
    print("--------------------------------")
    print(f"{label:10} | H={mean_h:5.1f}  S={mean_s:5.1f}  V={mean_v:5.1f}  Area={area:.0f}")

# =====================================
# FINAL RESULTS OVERLAY
# =====================================
cv2.putText(
    original,
    f"Rice: {rice_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)

cv2.putText(
    original,
    f"Foreign: {foreign_count}",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    2
)

print("\n===========================")
print("FINAL RESULT")
print("===========================")
print(f"Rice           : {rice_count}")
print(f"Foreign objects: {foreign_count}")

# Save result
cv2.imwrite("./test/result.jpg", original)
print("\nSaved → result.jpg")

# Show windows
cv2.imshow("Original", img)
cv2.imshow("Threshold", thresh)
cv2.imshow("Detection", original)

cv2.waitKey(0)
cv2.destroyAllWindows()