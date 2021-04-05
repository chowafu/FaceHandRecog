import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)
faceCascade = cv.CascadeClassifier("resource/haarcascade_frontalface_default.xml")
while True:
    success, img = vid.read()

    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 120], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    cv.drawContours(img, [contours], -1, (255,255,0), 2)

    hull = cv.convexHull(contours)
    cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)

    faces = faceCascade.detectMultiScale(blurred, 1.1, 6)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv.putText(img, "Face", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 6, cv.LINE_AA)
        cv.putText(img, "Face", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    if defects is not None:
        cnt = 0
    for i in range(defects.shape[0]): 
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) 
        if angle <= np.pi / 2: 
            cnt += 1
            cv.circle(img, far, 4, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt+1

    if cv.waitKey(25) & 0xFF ==ord('q'):
        break

    cv.imshow('final_result',img)