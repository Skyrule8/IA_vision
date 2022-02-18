"""import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# number signifies camera
cap = cv2.VideoCapture(0)


def substring_after(s, delim):
    return s.partition(delim)[2]


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex, ey, ew, eh) in eyes:
        #   cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        roi_gray2 = gray[ey:ey + eh, ex:ex + ew]
        roi_color2 = img[ey:ey + eh, ex:ex + ew]
        #   circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=1, minRadius=0,
                                   maxRadius=0)
        try:
            for i in circles[0, :]:
                print(i[2])

                foo = str(i[2])
                last = foo.split('.')[-1]
                int(last)
                # draw the outer circle
                cv2.circle(roi_color2, (i[0], i[1]), i[2], (255, 255, 255), 2)
                print("drawing circle")
                # draw the center of the circle
                cv2.circle(roi_color2, (i[0], i[1]), 2, (255, 255, 255), 3)
        except Exception as e:
            print(e)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #   cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
import cv2
import numpy as np

testKeypoints = False

# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector2 = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    global frame
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.1, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        cv2.rectangle(eyes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    #   Création de la trackbar pour "optimiser" le seuil
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, _frame = cap.read()
        #   Détection visage
        face_frame = detect_faces(_frame, face_cascade)
        if face_frame is not None:
            #   Dététection des yeux par rapport au visage
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                # Séparation de chaque oeil
                if eye is not None:
                    #   Récupération de la valeur de la trackbar
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    #   Enlevement (?) des cils
                    eye = cut_eyebrows(eye)
                    #   Calcul des points pour le contour des pupiles
                    keypoints = blob_process(eye, threshold, detector2)
                    if keypoints:
                        testKeypoints = True
                    #   Dessins en fonction des points
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', _frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
import math
import cv2

eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5
ret, frame = cap.read()
contours = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    for (x_eye, y_eye, w_eye, h_eye) in eyes:
        pupil_frame = gray[y_eye:y_eye + h_eye, x_eye:x_eye + w_eye]
        ret, thresh = cv2.threshold(pupil_frame, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow("threshold", thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)

        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            radius = 0.15 * (w + h)

            area_condition = (100 <= area <= 200)
            symmetry_condition = (abs(1 - float(w) / float(h)) <= 0.2)
            fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.4)
            cv2.circle(frame, (int(x_eye + x + radius), int(y_eye + y + radius)), int(1.3 * radius), (0, 180, 0), -1)

    cv2.imshow('Pupil Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret,frame = cam.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # rajoute un effet de flou

    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # afficher les 2 images
    cv2.imshow("Original image", frame)
    cv2.imshow("Blurred image", blurred_image)

    # affiche les lignes
    canny = cv2.Canny(blurred_image, 30, 100)
    cv2.imshow("Canny", canny)

    # trouve les contours
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()"""
