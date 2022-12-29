import cv2
import numpy as np

# Load YOLOv3 model and weights
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Load image
image = cv2.imread("image.jpg")
height, width, channels = image.shape

# Create input blob for the model
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

# Set the input for the model
net.setInput(blob)

# Run the model and get the output
output_layers = net.forward(net.getUnconnectedOutLayersNames())

# Loop through the detections and draw bounding boxes
for output in output_layers:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()