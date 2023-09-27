import cv2
from deepface import DeepFace

# Initialize video capture
cap = cv2.VideoCapture(0)
capture_interval = 2
capturing = True

# Function to analyze frame and display emotion
def analyze_frame():
    global capturing
    while capturing:
        ret, frame = cap.read()

        if ret:
            # Perform sentiment analysis
            result = DeepFace.analyze(frame, actions=['emotion'])

            # Check if any face is detected
            if len(result) > 0:
                dominant_emotion = result[0]['dominant_emotion']
            else:
                dominant_emotion = "No Face Detected"

            # Display emotion label
            cv2.putText(frame, f'Emotion: {dominant_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display video feed
            cv2.imshow('Emotion Analysis', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                capturing = False
                break

# Start analyzing frames
analyze_frame()

# Release video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
