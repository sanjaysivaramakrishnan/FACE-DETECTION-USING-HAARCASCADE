# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Overview

This project demonstrates face and eye detection using OpenCV's Haar Cascade Classifiers in both static images and real-time video. The implementation includes support for:
- Face detection in single-person and group photos
- Eye detection capabilities
- Real-time webcam face detection
- Different image resolutions testing

## Prerequisites

- Python 3.7 or above
- OpenCV (`opencv-python`)
- Matplotlib (`matplotlib`)
- Jupyter Notebook

Required files:
- `haarcascade_frontalface_default.xml`
- `haarcascade_eye.xml`

## Implementation Details

### I) Image Loading and Preprocessing

```python
import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('image_01.png', 0)  # Without glasses
img2 = cv2.imread('image_02.png', 0)  # With glasses
img3 = cv2.imread('image_03.png', 0)  # Group photo

# Resize for better detection
img1_resized = cv2.resize(img1, (1000,1000))
img2_resized = cv2.resize(img2, (1000,1000))
img3_resized = cv2.resize(img3, (1000,1000))
```

### II) Face Detection Implementation

The project uses two main detection functions:

1. Face Detection:
```python
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (127,0,255), 10)
    return face_img
```

2. Eye Detection:
```python
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eye(img):
    face_img = img.copy()
    face_rects = eye_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 2)
    return face_img
```

### III) Real-time Webcam Detection

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break
        
    result = detect_face(frame)
    cv2.imshow("Face Detection Through Webcam", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Notes

- The program includes testing of different image resolutions:
  - Original size
  - 1000x1000 resize
  - 2000x2000 resize
- Larger resolutions may provide better detection but require more processing power
- Real-time detection is optimized for webcam feed

## Usage Tips

1. For static images:
   - Use appropriate image sizes (1000x1000 recommended)
   - Test with both color and grayscale images
   - Ensure good lighting and clear faces

2. For webcam detection:
   - Ensure proper lighting
   - Press 'q' to exit the webcam feed
   - Keep face centered in frame

## Limitations and Future Improvements

- Detection accuracy may vary with:
  - Lighting conditions
  - Face angles
  - Presence of glasses/accessories
  - Image quality and resolution

Potential improvements:
- Add face recognition capabilities
- Implement multiple cascade classifiers
- Add support for profile face detection
- Optimize detection parameters
