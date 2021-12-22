import cv2
from djitellopy import tello
import time
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection


me = tello.Tello()
me.connect()
print("connecting to veronica")
print(f"current battery status -: {me.get_battery()}")
me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 20, 0)
time.sleep(.8)

p_time = 0
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
p_error = 0

def detect_face(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw face detections of each face.

        faces = []
        img_x, img_y, _ = image.shape

        if results.detections:
            for detection in results.detections:
                x = int(detection.location_data.relative_bounding_box.xmin * img_y)
                y = int(detection.location_data.relative_bounding_box.ymin * img_x)
                w = int(detection.location_data.relative_bounding_box.width * img_y)
                h = int(detection.location_data.relative_bounding_box.height * img_x)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                faces.append((x, y, w, h))
        return faces


def find_face(frame):
    faces = detect_face(frame)
    my_face_list_c = []
    my_face_list_area = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        c_x = x + w // 2
        c_y = y + h // 2
        cv2.circle(frame, (c_x, c_y), 2, (0, 255, 0), cv2.FILLED)
        my_face_list_c.append([c_x, c_y])
        my_face_list_area.append(w * h)
    if len(my_face_list_area) != 0:
        arg_max = my_face_list_area.index(max(my_face_list_area))
        return frame, [my_face_list_c[arg_max], my_face_list_area[arg_max]]
    else:
        return frame, [[0, 0], 0]


def track_face(info, w, pid, p_error):
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - p_error)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20

    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    me.send_rc_control(0, fb, 0, speed)
    # print(speed,fb)
    return error


while True:

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        frame = me.get_frame_read().frame
        frame = cv2.resize(frame, (w, h))

        frame, data = find_face(frame)
        p_error = track_face(data, w, pid, p_error)
        ctime = time.time()
        fps = 1 / (ctime - p_time)
        p_time = ctime
        cv2.putText(frame, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)
        # print("Area",data[1])

        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            me.land()
            break