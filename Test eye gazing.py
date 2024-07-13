import cv2

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

def detect_gaze(eyes):
    if len(eyes) != 2:
        return "Unknown"
    
    eye1, eye2 = eyes
    x1, y1, w1, h1 = eye1
    x2, y2, w2, h2 = eye2
    
    # Determine the relative positions of the eyes
    if x1 < x2:
        left_eye = eye1
        right_eye = eye2
    else:
        left_eye = eye2
        right_eye = eye1
    
    # Calculate the center of each eye
    left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
    right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)
    
    # Calculate the gaze direction based on the eye centers
    if left_eye_center[0] < right_eye_center[0] - 10:
        return "Looking Left"
    elif left_eye_center[0] > right_eye_center[0] + 10:
        return "Looking Right"
    else:
        return "Looking Straight"

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        # Draw rectangles around the detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Determine the gaze direction
        gaze_direction = detect_gaze(eyes)
        
        # Display the gaze direction on the frame
        cv2.putText(frame, gaze_direction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame with the detected faces and gaze direction
    cv2.imshow('Face and Eye Gaze Tracking', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()