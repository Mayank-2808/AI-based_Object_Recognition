#-- YOLO and OpenCV
import cv2
from ultralytics import YOLO
import numpy

#-- PyOpenGL
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

#-- Initialize GLFW 
glfw.init()
#-- GLFW window
# Window Width is 640 and Window Height is 480 (Same as that of YOLO, for proper object mapping)
window = glfw.create_window(640,480,"Detected Object(s) Graphical Representation",None, None)
glfw.set_window_pos(window=window,xpos = 400, ypos = 300)       
glfw.make_context_current(window=window)
glClearColor(1.0, 1.0, 0.0, 1.0);                           

#-- Function to convert Coordinates in Pixels to OpenGL usable coordinates
# x and y are cordinates in Pixels of detected objects obtained using YOLO and OpenCV
def pixel_to_opengl(x, y, window_width, window_height):
    return (2.0 * x) / window_width - 1.0, 1.0 - (2.0 * y) / window_height

#-- ARUCO MARKER
# Aruco Marker is used for calculating the detected object's physical dimension and pixels to centimeters ratio
#-- Load Aruco Marker
parms = cv2.aruco.DetectorParameters()
# Aruco Marker used in the project: DICT_5x5_50
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)  

#-- Access Webcam
cap = cv2.VideoCapture(0)   # 0 refers to webcam 

#-- Load the YOLOv8 Model
model = YOLO('yolov8n.pt')

#-- Capture as long as Webcam is available
while cap.isOpened():
    #-- Read from Webcam
    success, frame = cap.read()
    #-- GLFW Poll Events
    glfw.poll_events()
    #-- CLearing the GLFW/OpenGL Color Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    #-- if read from webcam is successful
    if success:
        #-- Detect Aruco Marker
        markerCorners, someInfo, someInfo_2 = cv2.aruco.detectMarkers(frame, arucoDict, parameters = parms)
        # markerCorners Output = [ Topleft (x,y), (TopRight (x,y), BottomRight (x,y) BottomLeft (x,y) ]
        
        #-- if Aruco Marker is present, do the distance and dimenstion calculation AND Plot OpenGL figure
        if markerCorners: 
            #-- Convert float type corners data into integer for plotting
            markerCorners = numpy.int0(markerCorners)       
            
            #-- Obtain Parameter of the detected marker in Pixels
            markerPerimeter = cv2.arcLength(markerCorners[0], True)   # 1st marker (since we are using only one marker)
            
            #-- Draw Boudary box around the detected marker   
            # ([44,166,255) = ORANGE Boundary Box around the Aruco Marker]
            cv2.polylines(frame,markerCorners, True, (44,166,244), 2)

            #-- Note: Aruco marker is a square of 5cm, 
            # So, the parameter (in px) of the marker corresponds to 20cm in the real world.
            
            #-- Calculating Pixel to cm ratio, to calculate height and width information
            # Divide width and height data (in Pixels) by px_cm ratio to convert them into centimeters
            px_cm = (markerPerimeter / 20)

            #-- Running YOLO on captured video frames
            # conf is set to 85% to reduce number of detected object 
            results = model.predict(frame, conf = 0.7)
            objects = results[0].boxes.cpu().numpy()
            
            #-- Number of object detected, to print individual dimension and distance information
            num_ObjectsDetected = len(objects.boxes)        

            #-- Iterate every Detected Object and get their data (xyxy, xywh = Center cordinates (x,y), Dimension (width, height))
            for data in objects.data:
                #-- Calculating Center of the detected object
                x_cor = objects.xywh[:,0]                       # Center x                                                  
                y_cor = objects.xywh[:,1]                       # Center y
                
                #-- Calculating Width and height of the detected object
                width_px = objects.xywh[:,2]                    # Width
                height_px = objects.xywh[:,3]                   # Height

                #-- Calculating Object Class 
                classID = objects.data[:,5]                     # Class ID

                #-- Iterate over every detected object to print their information (Center cordinates and Dimensions)
                for i in range (num_ObjectsDetected):
                    #-- Formating width and height to 2 decimal places
                    wi = round(width_px[i],2)
                    hi = round(height_px[i],2)  
                    
                    #-- Getting Object's Class ID
                    objectID = classID[i]
                    
                    #-- Mark center of the detected object with a circle  
                    # [(255,0,0) BLUE color]
                    cv2.circle(frame, (x_cor[i].astype(int), y_cor[i].astype(int)),5, (255,0,0), 2)

                    #-- Putting Height and Width infirmation in cm
                    # To convert dimensions from px to cm, divide Dimension in Pixels with pc_cm ratio
                    
                    #-- Printing Width (cm)
                    cv2.putText(frame, f'Width {"%.2f" % (wi/px_cm)} cm', ((x_cor[i].astype(int) - 20), (y_cor[i].astype(int))), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                    #-- Printing Height (cm)
                    cv2.putText(frame, f'Height {"%.2f" % (hi/px_cm)} cm', ((x_cor[i].astype(int) - 20), (y_cor[i].astype(int) - 25)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                    
                    #-- For PyOpenGL Shapes
                    # Get cordinates of the bounding box
                    boxCordinates = objects.xyxy[i].astype(int)
                    
                    X1 = boxCordinates[0]
                    Y1 = boxCordinates[1]
                    #X2 = boxCordinates[2]
                    #Y2 = boxCordinates[3]
                    
                    # for PyOPenGL [a,b,c,d] = 
                    # (a,b) = Bottom Left, (c,b) = Bottom Right, (c,d) = Top Right, (a,d) = Top Left        
                    
                    #-- for converting Pixels data into PyOPenGL's cordinate system
                    x_pixel, y_pixel = X1, Y1             #X1,Y1
                    width_pixel, height_pixel = wi, hi    #width_px,height_px
                    
                    #-- COnverting data here
                    x_opengl, y_opengl = pixel_to_opengl(x_pixel, y_pixel, 640, 480)
                    width_opengl, height_opengl = (2.0 * width_pixel) / 640, (2.0 * height_pixel) / 480
                    
                    #-- Plotting PyOPenGL Shapes (GL_QUADS referes to Quadrilateral)
                    glPointSize(5.0)
                    glBegin(GL_QUADS)
                    #-- Shape COlor is (1.0,0,0) = RED     
                    glColor3f(1.0, 0.0, 0.0)
                    #-- Cordinates Data for plottting Shape                    
                    glVertex2f(x_opengl, y_opengl)
                    glVertex2f(x_opengl + width_opengl, y_opengl)
                    glVertex2f(x_opengl + width_opengl, y_opengl - height_opengl)
                    glVertex2f(x_opengl, y_opengl - height_opengl)
                    glEnd() 
                    #-- End Plotting Shape (Quadrilateral)

                    # Measuring Distance of Desired Objects from the camera (webcam)
                    # To know Class IDs of of the different objects recognized by YOLO refer to the file "ClassID_ultralytics_datasets_coco.txt"
                    
                    if objectID == 41:      # Class ID of CUP
                        widthOf_cup_cm = 11
                        # Calculated using file calculate_Focalpoint.py
                        calculatedFocalPoint = 582.6818
                        # Distance
                        distance = (widthOf_cup_cm * calculatedFocalPoint) / wi
                        cv2.putText(frame, f'Distance {"%.2f" % (distance)} cm', ((x_cor[i].astype(int) - 20), (y_cor[i].astype(int) + 30)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                        #print("Distance to cup ", distance)      

                    elif objectID == 39:    # Class ID of Bottle
                        widthOf_bottle_cm = 4
                        # Calculated using file calculate_Focalpoint.py
                        calculatedFocalPoint = 626.2499
                        distance = (widthOf_bottle_cm * 626.2499) / wi
                        cv2.putText(frame, f'Distance {"%.2f" % (distance)} cm', ((x_cor[i].astype(int)- 20), (y_cor[i].astype(int) + 30)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                        #print("Distance to Bottle ", distance)

                    elif objectID == 47:    # Class ID of Apple
                        widthOf_apple_cm = 7
                        # Calculated using file calculate_Focalpoint.py
                        calculatedFocalPoint = 557.1428
                        distance = (widthOf_apple_cm * calculatedFocalPoint) / wi
                        cv2.putText(frame, f'Distance {"%.2f" % (distance)} cm', ((x_cor[i].astype(int)- 20), (y_cor[i].astype(int) + 30)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                        #print("Distance to Apple ", distance)
                    
                    # Adding Banana
                    elif objectID == 46:    # Class ID of Banana
                        widthOf_banana_cm = 20
                        # Calculated using file calculate_Focalpoint.py
                        calculatedFocalPoint = 267.5
                        distance = (widthOf_banana_cm * calculatedFocalPoint) / hi # Here, height data is needed instead of wi
                        cv2.putText(frame, f'Distance {"%.2f" % (distance)} cm', ((x_cor[i].astype(int)- 20), (y_cor[i].astype(int) + 30)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                        #print("Distance to Banaa ", distance)
                    
                    # Add more elif statements for more objects
                    

            # Plot (figure)
            detectedObject = results[0].plot()
            # Display Captured Video with detected objects
            cv2.imshow("Detection and Distance", detectedObject)
        else:
            print("NO Aruco Marker Detected...")
    
    # GLFW Swap Buffers
    glfw.swap_buffers(window) 
    
    # Loop every 1000ms
    cv2.waitKey(1000)
