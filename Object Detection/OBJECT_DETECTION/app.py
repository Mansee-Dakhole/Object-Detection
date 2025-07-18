import cv2
import numpy as np
import openai
import os
from ultralytics import YOLO

# Securely load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment variables

# Load class names from file
classNames = []
with open('objects.txt', 'r') as f:
    classNames = f.read().splitlines()

# Assign random colors to each class
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))


def get_chatgpt_response(object_name):
    """
    Get a description of the detected object using ChatGPT.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an object detection assistant."},
                {"role": "user", "content": f"What is {object_name}?"}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    except openai.error.OpenAIError as e:
        print(f"Error fetching ChatGPT response: {e}")
        return "No response available."


def train_yolo_model():
    """
    Train YOLOv8 on custom dataset if not already trained.
    """
    if not os.path.exists("runs/detect/train/weights/best.pt"):
        print("Starting YOLOv8 training on custom dataset...")
        model.train(data="custom.yaml", epochs=50, imgsz=640)
        print("Training complete!")
    else:
        print("Model already trained. Loading best model...")


# Train YOLOv8 model if needed
model = YOLO("yolov8n.pt")  # Load initial YOLO model
train_yolo_model()

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access webcam.")
        break

    # Run YOLOv8 inference on frame
    results = model(frame)

    # Plot detections with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Process detections for ChatGPT response
    for r in results:
        boxes = r.boxes.cpu().numpy()  # Extract bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
            conf = round(box.conf[0], 2)  # Confidence score
            cls_id = int(box.cls[0])  # Class ID

            if 0 <= cls_id < len(classNames):  # Validate class ID
                object_name = classNames[cls_id]
            else:
                object_name = "Unknown Object"

            # Get ChatGPT response for the detected object
            chatgpt_response = get_chatgpt_response(object_name)
            print(f"ChatGPT Response for {object_name}: {chatgpt_response}")

    # Show annotated frame with bounding boxes
    cv2.imshow("Real-Time Object Detection with YOLOv8", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

