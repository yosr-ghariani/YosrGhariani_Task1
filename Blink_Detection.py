#importing cv2 for webcam capturing , mediapipe for the eyes landmarks, numpy as a necessity to run mediapipe and time to detect the blinks in real time
import cv2
import mediapipe as mp
import time
import numpy as np
#numpy-2.2.1 opencv-python-4.10.0.84

#Initializing the Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1) #this program only works on one face

#Eye landmarks for left and right eyes using the most prominent indexes of mediapipe
LEFT_EYE_LANDMARKS = [33, 160, 159, 158, 144, 153, 145, 154, 155]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 386, 384, 380, 374, 373, 390]

#Function to calculate the Eye Aspect Ratio (EAR equation)
def calculate_ear(landmarks, eye_points, img_width,img_height):
    #Selection specific eye landmarks and normalizing them with the image frame
    eye=np.array([(landmarks[idx].x * img_width, landmarks[idx].y * img_height) for idx in eye_points])

    #Calculating distances
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizental = np.linalg.norm(eye[0] - eye[3])
    #Calculating EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizental)
    return ear



#Constants for blinking detection
EAR_THRESHOLD = 0.6 # EAR below 0.6 is considered a blink in this project
CONSEC_FRAMES = 3 #Number of frames for a blink to be registered

#Initializing counters
blink_count = 0
frame_counter = 0

#OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #to adjust the video frame

start_time = time.time()
duration = 30 #taking an example of a 30-second window to count blinks (you can change it)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break #if a frame isn't detected

    frame = cv2.flip(frame, 1) #mirror the frame

    #Converting to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    #Drawing landmarks and processing EAR
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            #Calculating EAR for both eyes (with an average)
            img_height, img_width, _ = frame.shape
            left_ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS, img_width, img_height)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_LANDMARKS, img_width, img_height)

            avg_ear = (left_ear + right_ear) / 2.0

            #Checking if EAR is below the blink threshold
            if avg_ear < EAR_THRESHOLD: 
                frame_counter +=1
            else: 
                if frame_counter >= CONSEC_FRAMES:
                    blink_count +=1
                    print("blink detected, total blink: ", blink_count)
                frame_counter = 0

            #Draw EAR and real-time blink count on the frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            eye_coords = [(int(landmarks[idx].x * frame.shape[1]), int(landmarks[idx].y * frame.shape[0])) for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS]
            for coord in eye_coords:
                cv2.circle(frame, coord, 2, (255, 0, 255), -1)


       
    #Displaying the frame
    cv2.imshow('Blink Detection', frame)

    #Breaking after the defined duration (30s)
    if time.time() - start_time > duration:
        break
    #Or quitting with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print("Total blinks detected in", duration, "seconds: ", blink_count)









