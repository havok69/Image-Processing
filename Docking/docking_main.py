import cv2
import numpy as np

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