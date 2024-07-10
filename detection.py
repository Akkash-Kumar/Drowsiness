from scipy.spatial import distance    #facial data points distance calculation

from imutils import face_utils   #extract facial data points

import imutils  #resize

import dlib  #face detector and landmark detector loading

import cv2   #opencv

import winsound   #alarm sound

#alarm sound properties

frequency = 2500

duration = 1000   #end


def eyeAspectRatio(eyes):  #function definition

    #vertical eye distance
    A = distance.euclidean(eyes[1],eyes[5])   #up  

    B = distance.euclidean(eyes[2],eyes[4])   #down

    #Horizontal eye distance

    C = distance.euclidean(eyes[0],eyes[3])   

    e = (A + B) / (2.0 * C)

    return e




count = 0

eyeThresh = 0.3  #distance between vertical eyes coordinate(if it goes below this value then alarm beeps)

eyeFrames = 50 #duration of eye closure

shapePredictor = 'shape_predictor_68_face_landmarks.dat'   #file initialise


camera = cv2.VideoCapture(0)   #initialise primary camera

detector = dlib.get_frontal_face_detector()   #loading haar cascade frontal face algorithm for face detection from dlib library

predictor = dlib.shape_predictor(shapePredictor)   #loading facial landmark detector algorithm 

(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']   #extract data points of left eye

(rStart,rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']    #extract data points of right eye


while True:   #to run camera infinitely

    _,frame = camera.read()  #read frame from camera

    frame = imutils.resize(frame,width = 450)    #resize frame

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #convert grayscale img

    rects = detector(gray,0)    #to detect face coordinates

    for rect in rects:

        shape = predictor(gray,rect)   #to apply facial data points using face coordinates

        shape = face_utils.shape_to_np(shape)   #to array


        leftEye = shape[lStart:lEnd]   #to get left eye

        rightEye = shape[rStart:rEnd]   #to get right eye


        lEye = eyeAspectRatio(leftEye)   #function call
        
        rEye = eyeAspectRatio(rightEye)   #function call

        avgRatio = (lEye + rEye)/2.0   #average of two aspect ratio

        #to outline eyes by red color

        leftEyeHull = cv2.convexHull(leftEye)

        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame,[leftEyeHull],-1,(0,0,255),2)

        cv2.drawContours(frame,[rightEyeHull],-1,(0,0,255),2)  #end

        if avgRatio < eyeThresh:  #ratio less than threshold value

            count = count + 1

            if count >= eyeFrames:

                cv2.putText(frame,"DROWSINESS DETECTED",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   #put text

                winsound.Beep(frequency,duration)  #sound beeps

        else:

            count =0

    cv2.imshow('frame',frame)  #display camera

    key = cv2.waitKey(10)  #wait for 10 frames

    if key == 27:  #esc key is pressed,exit camera

        break

camera.release()  #release camera

cv2.destroyAllWindows()  #close window

    


        

    



