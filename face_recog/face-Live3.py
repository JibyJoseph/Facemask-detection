import face_recognition
import cv2
import pickle
import time

Encodings = []
Names = []
font = cv2.FONT_HERSHEY_SIMPLEX
fpsReport = 0
scaleFactor = 0.33
with open('/home/kiranjoy/pyPro/face_recog/train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

cam = cv2.VideoCapture(0)

while(True):
    timeStamp = time.time()
    _, frame = cam.read()
    frameSmall = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
    # frame = cv2.resize(frame, (480, 320))
    framergb = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    facePos = face_recognition.face_locations(framergb, model='cnn')
    allEncodings = face_recognition.face_encodings(framergb, facePos)
    for Pos, face_encoding in zip(facePos, allEncodings):
        name = 'unknown'
        (top, right, bottom, left) = Pos
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
        top = int(top/scaleFactor)
        right = int(right/scaleFactor)
        bottom = int(bottom/scaleFactor)
        left = int(left/scaleFactor)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top-6),
                    font, .75, (255, 0, 255), 2)
    dt = time.time()-timeStamp
    fps = 1/dt
    print('fps: ', fps)
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
