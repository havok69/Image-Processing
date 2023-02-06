import cv2

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries of the white color
    lower_white = (0, 0, 200)
    upper_white = (180, 30, 255)

    # Create a mask with the white color boundaries
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply the mask to the frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the result
    cv2.imshow("Result", res)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()