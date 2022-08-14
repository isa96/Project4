from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import time

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

maskNet = load_model("mask_detector.model")

def main():

    cap = cv2.VideoCapture(0)

    pTime = time.time()
    while True:
        _, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate (results.detections):
                bboxC = detection.location_data.relative_bounding_box
                x,y,w,h = rectanglePosition(img, bboxC)

                preds = predictMask(imgRGB, x,y,w,h, maskNet)
                if preds < 0.5:
                    label = "Mask"
                    color = (0,255,0)
                else:
                    label = "No Mask"
                    color = (0,0,255)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                img = fancyRectangle(img, x,y,w,h, color)

        img, pTime = showFPS(img, pTime)
        cv2.imshow("img", img)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()        
    cv2.destroyAllWindows()
    

def predictMask(imgRGB, x,y,w,h, maskNet):
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    imgFaces = []

    imgFace = imgRGB[y:y+h,x:x+w]
    imgFace = cv2.resize(imgFace, (224, 224))
    imgFace = img_to_array(imgFace)
    imgFace = preprocess_input(imgFace)
    imgFaces.append(imgFace)

    imgFaces = np.array(imgFaces)
    preds = maskNet.predict(imgFaces, batch_size=32)
    return preds[0]

def fancyRectangle(img, x,y,w,h, color, l=30, t=3):
    x1,  y1 = x+w, y+h

    cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
    # Top Left x,y
    cv2.line(img, (x,y), (x+l, y), color, t)
    cv2.line(img, (x,y), (x, y+l), color, t)
    # Top Right x1,y
    cv2.line(img, (x1,y), (x1-l, y), color, t)
    cv2.line(img, (x1,y), (x1, y+l), color, t)
    # Bottom Left x,y1
    cv2.line(img, (x,y1), (x+l, y1), color, t)
    cv2.line(img, (x,y1), (x, y1-l), color, t)
    # Bottom Roght x1,y1
    cv2.line(img, (x1,y1), (x1-l, y1), color, t)
    cv2.line(img, (x1,y1), (x1, y1-l), color, t)
    return img

def rectanglePosition(img, bboxC):
    h,w,c = img.shape
    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
        int(bboxC.width * w), int(bboxC.height * h)
    return bbox

def showFPS(img, pTime):
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    fps = round(fps,2)
    fps = "FPS: " + str(fps)
    cv2.putText(img, fps, (5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    return img, cTime

if __name__ == "__main__":
    main()