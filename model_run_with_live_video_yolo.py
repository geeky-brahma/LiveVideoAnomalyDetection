import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Class labels
classes = {
    0: 'Abuse', 1: 'Arrest', 2: 'Arson', 3: 'Assault', 4: 'Burglary',
    5: 'Explosion', 6: 'Fighting', 7: 'Normal Videos', 8: 'RoadAccidents',
    9: 'Robbery', 10: 'Shooting', 11: 'Shoplifting', 12: 'Stealing', 13: 'Vandalism'
}

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Load MobileNet model
mobilenet_model = load_model('mobilenet_anomaly_model_final.h5')

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

def classify_object(image):
    processed_image = preprocess_image(image)
    preds = mobilenet_model.predict(processed_image)
    label_index = np.argmax(preds)
    label = classes.get(label_index, "Unknown")
    return label

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform YOLO object detection
    results = yolo_model(frame)
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            obj_crop = frame[y1:y2, x1:x2]
            
            if obj_crop.size > 0:
                label = classify_object(obj_crop)
                print("Detected object:", label)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display output
    cv2.imshow('YOLO + MobileNet', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
