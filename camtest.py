import cv2

devices = [
    0, 1, 2, 3,
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3"
]

for dev in devices:
    print(f"\nTrying: {dev}")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("  → Failed to open")
        continue
        
    ret, frame = cap.read()
    if ret:
        print(f"  → SUCCESS! Shape: {frame.shape}")
        cv2.imwrite(f"test_{str(dev).replace('/', '_')}.jpg", frame)
        print(f"  → Saved test image")
    else:
        print("  → Opened but cannot read frame")
    
    cap.release()