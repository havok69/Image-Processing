import cv2
import numpy as np
import depthai
import blobconverter


pipeline = depthai.Pipeline()
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
# Set path of the blob (NN model). We will use blobconverter to convert&download the model
# detection_nn.setBlobPath("/path/to/model.blob")
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

cam_rgb.preview.link(detection_nn.input)

xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

with depthai.Device(pipeline) as device:
    #device = depthai.Device(pipeline, usb2Mode=True)

    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    frame = None
    detections = []

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

while True:
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()
    if in_rgb is not None:
        frame = in_rgb.getCvFrame()
    if in_nn is not None:
        detections = in_nn.detections
    if frame is not None:
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("preview", frame)


# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Threshold the frame to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find the contours of the white objects in the frame
    contours, hierarchy = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assumed to be the docking station)
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the contour on the frame
    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

    # Find the center of the contour
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw the center of the contour on the frame
    cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

    # Send commands to the microcontroller to center the ROV on the docking station
    if cX < frame.shape[1] / 2 - 20:
        print("left")  # sending a left command to the microcontroller
    elif cX > frame.shape[1] / 2 + 20:
        print("right")  # sending a right command to the microcontroller
    else:
        print("forward")  # sending a forward command to the microcontroller

    # Find the height of the docking station
    height, width, channels = frame.shape
    distance = height - cY
    if distance < height / 2 - 20:
        print("ascend")  # sending an ascend command to the microcontroller
    elif distance > height / 2 + 20:
        print("descend")  # sending a descend command to the microcontroller
    else:
        print("hold")  # sending a hold command to the microcontroller

    # Show the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask_white)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()