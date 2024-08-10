import cv2
from ultralytics import YOLO
import torch

# Object's Known Properties
widthOfObject_cm = 20               
distanceOfObject_cm = 50            # from the camera  

# Class ID of the object
objectToDetecte_ClassID = 46        # Refer to the ClassID_ultralytics_datasets_coco.txt

# To calcuate Focal Point once in the while True loop
iterator = 1

# Access webcam
cap = cv2.VideoCapture(0)   # 0 refers to webcam 

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')


while cap.isOpened():

    # Read from webcam
    success, frame = cap.read()

    # if Read from webcam is successful
    if success:
        # Run YOLO on captured video (object is frame) AND  Return detected objects with confidence of >=80% 
        # to reduce detected objects
        results = model.predict(frame, conf = 0.75)

        # .cpu().numpy() for easy calculations
        objectClass = results[0].boxes.cls
        objectxyxy = results[0].boxes.cpu().numpy().xyxy    # Boundary Box Coordinates XYXY
        objectxywh = results[0].boxes.cpu().numpy().xywh    # Boundary box Coordinated XY + w (width) and h (hight)

        # Detected Objects Class ID
        output = torch.tensor(objectClass)
        
        #detected object
        detectedObject = results[0].plot()

        # Check for Objects and Calculate Focal Point 
        # Checking for Desired Obejct
        if objectToDetecte_ClassID in output:   
            print("----------------------------------------------------------------------")
            print(f'Desired Object of class ID {objectToDetecte_ClassID} is Detected')
            # Calculating Width of the detected Object in Pixels
            widthOfObject_px = objectxywh[0][3].astype(int)
            print("width", widthOfObject_px)
            # Calculating Focal Point ONCE        
            focalPoint = (widthOfObject_px * distanceOfObject_cm) / widthOfObject_cm
            print("Calculated Focal Point for the Object is ", focalPoint)
            
            # ONCE, focal point is calculated [ONLY ONCE], that value will be used for distance
            while iterator:
                calculatedFocalPoint = focalPoint
                iterator = 0
                print("Focal Point for distance measurement is computed")                
            
            print("Focal Point (Fixed Value) for this object is ",calculatedFocalPoint)
            # Distance Calculations
            distance = (widthOfObject_cm * calculatedFocalPoint ) / widthOfObject_px
            print("----------------------------------------------------------------------")
            
            # Distance Text
            cv2.putText(detectedObject, f'Distance: {round(distance,2)} cm', (10,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),2)
            cv2.putText(detectedObject, f'Focal Point: {(calculatedFocalPoint)}', (30,55), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255),2)    

        # Display Captured Video with detected objects
        cv2.imshow("Detection, Focal Point and Distance", detectedObject)

    # loop every 1000ms (main loop: while)  
    cv2.waitKey(1000)
