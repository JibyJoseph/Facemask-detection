import face_recognition
import cv2
import os
import pickle

font = cv2.FONT_HERSHEY_SIMPLEX

Names = []
Encodings = []

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)


unKnownFaces = '/home/kiranjoy/pyPro/face_recog/demoImages/unknown/'
for file in os.listdir(unKnownFaces):

    testImage = face_recognition.load_image_file(
        os.path.join(unKnownFaces, file))

    facePositions = face_recognition.face_locations(testImage)
    allEncodings = face_recognition.face_encodings(testImage, facePositions)

    testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

    for Pos, face_encoding in zip(facePositions, allEncodings):
        name = 'unknown'
        (top, right, bottom, left) = Pos
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
        cv2.rectangle(testImage, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(testImage, name, (left, top-6),
                    font, .75, (255, 0, 255), 2)
    cv2.imshow('frame', testImage)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
    break
