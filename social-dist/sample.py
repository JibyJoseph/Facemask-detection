import jetson.inference
import jetson.utils
#import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#net = jetson.inference.detectNet("pednet", threshold=0.5)
camera = jetson.utils.videoSource('/dev/video0')      # '/dev/video0' for V4L2,
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	for detection in detections:
		print(detection)
	#cv2.imshow("Frame", img)
	#cv2.moveWindow('Frame',100,100)
	#key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	print("break")
	#	break
	print('##########################')
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
