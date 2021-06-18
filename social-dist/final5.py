import jetson.inference
import jetson.utils
import cv2
from scipy.spatial import distance as dist
import numpy as np
import time
import yaml,imutils

def compute_perspective_transform(corner_points,width,height,image):
	""" Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    """
	# Create an array out of the 4 corner points
	corner_points_array = np.float32(corner_points)
	# Create an array with the parameters (the dimensions) required to build the matrix
	img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
	# Compute and return the transformation matrix
	matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
	#img_transformed = cv2.warpPerspective(image,matrix,(width,height))
	return matrix


def compute_point_perspective_transformation(matrix,list_downoids):
	""" Apply the perspective transformation to every ground point which have been detected on the main frame.
    @ matrix : the 3x3 matrix 
    @ list_downoids : list that contains the points to transform
    return : list containing all the new points
    """
	# Compute the new coordinates of our points
	list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
	transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
	# Loop over the points and add them to the list that will be returned
	transformed_points_list = list()
	for i in range(0,transformed_points.shape[0]):
		transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
	return transformed_points_list



#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
net = jetson.inference.detectNet("pednet", threshold=0.5)
#net = jetson.inference.detectNet("multiped", threshold=0.5)
#camera = jetson.utils.videoSource('/dev/video0')      # '/dev/video0' for V4L2,
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
#cap = cv2.VideoCapture("pedestrians3.mp4")
cap = cv2.VideoCapture("ped6.mp4")
#cap = cv2.VideoCapture("PETS2009.mp4")
#cap = cv2.VideoCapture("vid_short.mp4")
min_distance = 800
font = cv2.FONT_HERSHEY_SIMPLEX
dtav = 0


print("[ Loading config file for the bird view transformation ] ")
with open("conf/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	size_frame = cfg["image_parameters"]["size_frame"]
print(" Done : [ Config file loaded ] ...")

ret, img = cap.read()
img = imutils.resize(img, width=800)
h,w,_ = img.shape
matrix = compute_perspective_transform(corner_points,w,h,img)


while cap.isOpened():
	start_time = time.time()
	ret, img = cap.read()
	img = imutils.resize(img, width=800)
	if not ret:
		break
	bgr_img = jetson.utils.cudaFromNumpy(img)
	rgb_img = jetson.utils.cudaAllocMapped(width=bgr_img.width,height=bgr_img.height,format='rgb8')
	jetson.utils.cudaConvertColor(bgr_img, rgb_img)
	dets = net.Detect(rgb_img) #detections
	violations = set()
	birds_eye = np.zeros((3000,4000,3), np.uint8)
	for det in dets:
		if det.ClassID != 0:
			dets.remove(det)
	#for det in dets:
	#	print(detection)
		#img = cv2.rectangle(img, (int(det.Left),int(det.Top)), (int(det.Right),int(det.Bottom)),(255,0,0),1)
		#print(det.Left,det.Top, det.Right,det.Bottom)
	'''
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
	'''	

	if len(dets) >= 2:
		centroids = np.array([det.Center for det in dets])
		centroids = compute_point_perspective_transformation(matrix,centroids)
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
			cen = (int(centroids[i][0]+1000), int(centroids[i][1]+1000))
			birds_eye = cv2.circle(birds_eye, cen, 30, color, -1)
			birds_eye = cv2.circle(birds_eye, cen, int(min_distance/2+6), color, 12)



	dt = time.time()-start_time
	dtav = .9*dtav + .1*dt
	fps = 1/dtav
	cv2.putText(img, "fps: "+ str(round(fps,1)), (0, 15), font, .5, (155, 0, 155), 2)

	cv2.imshow("Frame", img)
	cv2.moveWindow('Frame',0,0)
	cv2.imshow("Birds eye view", imutils.resize(birds_eye, width = 800))
	cv2.moveWindow('Birds eye view',800,0)
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
