import face_recognition
import os
import pickle

Names = []
Encodings = []
knownFaces = '/home/kiranjoy/pyPro/face_recog/demoImages/faceDataset/'
for file in os.listdir(knownFaces):
    Names.append(file[:-4])
    image = face_recognition.load_image_file(os.path.join(knownFaces, file))
    Encodings.append(face_recognition.face_encodings(image)[0])

with open('train.pkl', 'wb') as f:
    pickle.dump(Names, f)
    pickle.dump(Encodings, f)
