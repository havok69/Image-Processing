import cv2
import numpy as np
import serial # for serial communication with the microcontroller

# Connect to the microcontroller
ser = serial.Serial('COM3', 9600) # change the port and baudrate accordingly

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the frame to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND the mask with the original frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Count the number of non-zero pixels in the mask
    count = np.count_nonzero(mask)

    # If the count is above a certain threshold, stop the thrusters
    if count > 1000:
        ser.write(b'stop\n') # sending a stop command to the microcontroller
    else:
        ser.write(b'continue\n') # sending a continue command to the microcontroller

    # Show the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()