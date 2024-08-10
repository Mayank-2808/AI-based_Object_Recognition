import cv2
from ultralytics import YOLO
import numpy

# -- Focal Length calculation
# Aruco Marker is used for calculating the detected object's physical dimension and pixels to centimeters ratio

# Load Aruco Marker
parms = cv2.aruco.DetectorParameters()
# Aruco Marker used in the project: DICT_5x5_50
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)  

# Access Webcam
cap = cv2.VideoCapture(0)   # 0 refers to webcam 

# Load the YOLOv8 Model
model = YOLO('yolov8n.pt')

# Capture as long as Webcam is available
while cap.isOpened():
    # Read from Webcam
    success, frame = cap.read()
    # if read is successful
    if success:
        # Detect Aruco Marker
        markerCorners, someInfo, someInfo_2 = cv2.aruco.detectMarkers(frame, arucoDict, parameters = parms)
        
        # if Aruco Marker is present, do the distance and dimenstion calculation
        if markerCorners: 
            # Convert float type corners data into integer for plotting
            markerCorners = numpy.int0(markerCorners)       # markerCorners = [ Topleft (x,y), (TopRight (x,y), BottomRight (x,y) BottomLeft (x,y) ]
            
            # Obtain Parameter of the detected marker in px
            markerPerimeter = cv2.arcLength(markerCorners[0], True)   # 1st marker (since we are using only one marker)
            
            # Draw Boudary box around the detected marker   [ORANGE Boundary Box]
            cv2.polylines(frame,markerCorners, True, (44,166,244), 2)

            # Note: Aruco A marker is a square of 5cm, 
            # so the parameter (in px) of the marker corresponds to 20cm in the real world.
            
            # Calculating Pixel to cm ratio, to calculate height and width information
            px_cm = (markerPerimeter / 20)

            # Running YOLO on captured video frames [conf is confidence score]
            results = model.predict(frame, conf = 0.7)
            objects = results[0].boxes.cpu().numpy()
            
            # Number of object detected, to print individual dimension and distance information
            num_ObjectsDetected = len(objects.boxes)        

            for data in objects.data:
                # Calculating Center of the detected object
                x_cor = objects.xywh[:,0]   #xywh = (x,y) (width,height)
                y_cor = objects.xywh[:,1]
                
                # Calculating Width and height of the detected object
                width_px = objects.xywh[:,2]
                height_px = objects.xywh[:,3]

                # Calculating Object Class 
                classID = objects.data[:,5]     
                
                # Iterate over every detected object
                for i in range (num_ObjectsDetected):
                    # Formating width and height to 2 decimal places
                    wi = round(width_px[i],2)
                    hi = round(height_px[i],2)  
                    
                    # Getting Object's Class ID
                    objectID = classID[i]
                    # Mark center of the detected object with a circle  [BLUE color]
                    cv2.circle(frame, (x_cor[i].astype(int), y_cor[i].astype(int)),5, (255,0,0), 2)

                    # Putting Height and Width data in cm
                    # To convert dimensions from px to cm, divide px dimension with pc_cm ratio
                    cv2.putText(frame, f'Width {"%.2f" % (wi/px_cm)} cm', ((x_cor[i].astype(int) - 20), (y_cor[i].astype(int))), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
                    cv2.putText(frame, f'Height {"%.2f" % (hi/px_cm)} cm', ((x_cor[i].astype(int) - 20), (y_cor[i].astype(int) - 25)), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)

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
                    
                    # Add more elif statements for more objects 

            
            # Plot (figure)
            detectedObject = results[0].plot()
            # Display Captured Video with detected objects
            cv2.imshow("Detection and Distance", detectedObject)
        else:
            print("NO Aruco Marker Detected...")

    # Loop every 1000ms
    cv2.waitKey(1000)  
