import cv2
import numpy as np
import yaml
import imutils

 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])



#video_name = input("Enter the exact name of the video (including .mp4 or else) : ")
video_name = "ped6.mp4"
size_frame = 800

vs = cv2.VideoCapture(video_name)

(frame_exists, img) = vs.read()
img = imutils.resize(img, width=int(size_frame))

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)

# Get the size of the image for the calibration
width,height,_ = img.shape

# Create an empty list of points for the coordinates
list_points = list()

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)


if __name__ == "__main__":
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == 4:
            # Return a dict to the YAML file
            config_data = dict(
                image_parameters = dict(
                    p2 = list_points[3],
                    p1 = list_points[2],
                    p4 = list_points[0],
                    p3 = list_points[1],
                    width_og = width,
                    height_og = height,
                    size_frame = size_frame,
                    ))
            # Write the result to the config file
            with open('conf/config_birdview.yml', 'w') as outfile:
                yaml.dump(config_data, outfile, default_flow_style=False)
            break
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
