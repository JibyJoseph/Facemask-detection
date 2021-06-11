from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import face_recognition
import pickle


# In[2]:
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


# In[3]:
Encodings = []
Names = []
font = cv2.FONT_HERSHEY_SIMPLEX
fpsReport = 0
dtav=0

with open('/home/kiranjoy/pyPro/face_recog/train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


# In[4]:


def GetLabelandColor(improperMask, mask, withoutMask):
    l=[improperMask, mask, withoutMask]
    n = np.argmax(l)
    if n==0:
        return "ImproperMask",(255, 0, 0)
    elif n==1:
        return "Mask",(0, 255, 0)
    else:
        return "NoMask",(0, 0, 255)


# In[5]:


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	timeStamp = time.time()
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame = cv2.flip(frame,1)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(improperMask, mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label,color = GetLabelandColor(improperMask, mask, withoutMask)

		if label == 'NoMask':
			encode = face_recognition.face_encodings(frame, [(startY,endX,endY,startX)])
			name='unknown'
			matches = face_recognition.compare_faces(Encodings, encode[0])
			if True in matches:
				first_match_index = matches.index(True)
				name = Names[first_match_index]
			cv2.putText(frame, name, (startX, endY+15), font, .75, (255, 0, 255), 2)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(improperMask, mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),font, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	dt = time.time()-timeStamp
	dtav = .9*dtav + .1*dt
	fps = 1/dtav
	cv2.putText(frame, "fps: "+ str(round(fps,1)), (0, 15), font, .5, (155, 0, 155), 2)
	# show the output frame
	#frame = cv2.resize(frame,(800,600),)
	cv2.imshow("Frame", cv2.resize(frame,(800,600),))
	cv2.moveWindow('Frame',100,100)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("break")
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
