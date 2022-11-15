import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose() as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        #modificar fotogramas
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        #asignacion de valores del vector
        if results.pose_landmarks is not None:
            #print(int(results.pose_landmarks.landmark[20].x * width))
            x1 = int(results.pose_landmarks.landmark[20].x * width)
            y1 = int(results.pose_landmarks.landmark[20].y * height)
            x2 = int(results.pose_landmarks.landmark[19].x * width)
            y2 = int(results.pose_landmarks.landmark[19].y * height)
            x3 = int(results.pose_landmarks.landmark[7].x * width)
            y3 = int(results.pose_landmarks.landmark[7].y * height)
            x4 = int(results.pose_landmarks.landmark[8].x * width)
            y4 = int(results.pose_landmarks.landmark[8].y * height)
            x5 = int(results.pose_landmarks.landmark[13].x * width)
            y5 = int(results.pose_landmarks.landmark[13].y * height)
            x6 = int(results.pose_landmarks.landmark[14].x * width)
            y6 = int(results.pose_landmarks.landmark[14].y * height)

            #vectores
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])
            p4 = np.array([x4, y4])
            p5 = np.array([x5, y5])
            p6 = np.array([x6, y6])

            l1 = np.linalg.norm(p1 - p2)
            l2 = np.linalg.norm(p1 - p4)
            l3 = np.linalg.norm(p2 - p3)
            l4 = np.linalg.norm(p3 - p5)
            l5 = np.linalg.norm(p4 - p6)
            #print(l1) 

            #Visualizacion
            cv2.line(frame ,p1 ,p2 ,(255, 255, 255), 2)
            #cv2.putText(frame, str(int(l1)),p2-p1, 1, 1.5, (128, 0, 250), 2) pendiente...
            cv2.line(frame ,p1 ,p4 ,(255, 255, 255), 2)
            cv2.line(frame ,p2 ,p3 ,(255, 255, 255), 2)
            cv2.line(frame ,p3 ,p5 ,(255, 255, 255), 2)
            cv2.line(frame ,p4 ,p6 ,(255, 255, 255), 2)

            cv2.circle(frame, p1, 6, (0, 0, 255), -1)
            cv2.circle(frame, p2, 6, (0, 0, 255), -1)
            cv2.circle(frame, p3, 6, (255, 104, 0), -1)
            cv2.circle(frame, p4, 6, (255, 104, 0), -1)
            cv2.circle(frame, p5, 6, (0, 255, 114), -1)
            cv2.circle(frame, p6, 6, (0, 255, 114), -1)

            """
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            """
        #manipulacion del frame
        cv2.imshow("Frame", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()