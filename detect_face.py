import cv2
import threading
from deepface import DeepFace

# Define Camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define proportions
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

ref_img = cv2.imread("ref_img.jpg")

# Using DeepFace check if the faces are the same
def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, ref_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret, frame = capture.read()

    if ret:
        if counter % 30 == 0:
            try: 
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # If there is a match write the "Match" green otherwise write "No match" red
        if face_match:
            cv2.putText(frame, "Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()