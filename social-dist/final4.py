import jetson.inference
import jetson.utils
import cv2
from scipy.spatial import distance as dist
import numpy as np
import time

#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
net = jetson.inference.detectNet("pednet", threshold=0.5)
#net = jetson.inference.detectNet("multiped", threshold=0.5)
#camera = jetson.utils.videoSource('/dev/video0')      # '/dev/video0' for V4L2,
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
cap = cv2.VideoCapture("pedestrians2.mp4")
#cap = cv2.VideoCapture("one.mp4")
#cap = cv2.VideoCapture("vid_short.mp4")
min_distance = 250
font = cv2.FONT_HERSHEY_SIMPLEX
dtav = 0
while cap.isOpened():
	start_time = time.time()
	ret, img = cap.read()
	if not ret:
		break
	bgr_img = jetson.utils.cudaFromNumpy(img)
	rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,height=bgr_img.height,format='rgb8')
	jetson.utils.cudaConvertColor(bgr_img, rgb_img)
	dets = net.Detect(rgb_img) #detections
	violations = set()
	for det in dets:
		if det.ClassID != 0:
			dets.remove(det)
	#for det in dets:
	#	print(detection)
		#img = cv2.rectangle(img, (int(det.Left),int(det.Top)), (int(det.Right),int(det.Bottom)),(255,0,0),1)
		#print(det.Left,det.Top, det.Right,det.Bottom)
	if len(dets) >= 2:
		centroids = np.array([det.Center for det in dets])
		distMatrix = dist.cdist(centroids, centroids, metric='euclidean')
		for i in range(0, distMatrix.shape[0]):
			for j in range(i+1, distMatrix.shape[1]):
				if distMatrix[i, j] < min_distance:
					violations.add(i)
					violations.add(j)
	
	for (i, det) in enumerate(dets):
		color = (0,255,0)
		if i in violations:
			color = (0,0,255)
		img = cv2.rectangle(img, (int(det.Left),int(det.Top)), (int(det.Right),int(det.Bottom)),color,1)
		img = cv2.putText(img,str(round(det.Confidence,2)),(int(det.Left),int(det.Top)), font, 1, color, 3)
	

	dt = time.time()-start_time
	dtav = .9*dtav + .1*dt
	fps = 1/dtav
	cv2.putText(img, "fps: "+ str(round(fps,1)), (0, 15), font, .5, (155, 0, 155), 2)

	cv2.imshow("Frame", img)
	cv2.moveWindow('Frame',100,100)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("break")
		break
	#print('##########################')
	#display.Render(img)
	#display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
cap.release()
cv2.destroyAllWindows()
