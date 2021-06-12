import jetson.inference
import jetson.utils
import cv2

cap = cv2.VideoCapture('vid_short.mp4')

# net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
net = jetson.inference.detectNet("pednet", threshold=0.5)
#camera = jetson.utils.videoSource('/dev/video0')      # '/dev/video0' for V4L2,
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
    ret,frame = cap.read()
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_frame = jetson.utils.cudaFromNumpy(frame_rgba)
    detections = net.Detect(cuda_frame)
    for detection in detections:
        print(detection)
    display.Render(cuda_frame)
    #frame = jetson.utils.cudaToNumpy(cuda_frame)
    #cv2.imshow('frame',frame)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
cap.release()
cap.destroyAllWindows()
