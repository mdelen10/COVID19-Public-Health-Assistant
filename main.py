# webcam_people.py

import os
import cv2
import sys
import time
import numpy
import math
import random
import subprocess
from multiprocessing import Process

blue= (255,0,0)
green= (0,255,0)
red= (0,0,255)
yellow = (0,255,255)

cascPath = "haarcascade_frontalface_default.txt"
NUM_WIDTHS = 6.5
WAIT = 1 # ms

def dist(x,y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

def depth_far(fx, fy):
    ratio = fx.w / fy.w
    if ratio <= 0.35 or ratio >= 1/.35:
        return True
    return False

def combinations(faces):
    out=set()
    groups = list()

    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            fi=faces[i]
            fj=faces[j]
            distance=  fi.distance(fj)
            avg_distance= ( fi.w + fj.w )/2
            safe_distance= NUM_WIDTHS * avg_distance
            
            if distance < safe_distance and (not depth_far(fi, fj)):
                out.add(fi)
                out.add(fj)
                groups.append((fi,fj))
    return out, groups     


class Face():
    def __init__(self, tup, idx):
        (x,y,w,h) = tup
        self.x0 = x
        self.y0 = y
        self.x1 = self.x0 + w
        self.y1 = self.y0 + h
        self.w = w
        self.h = h
        # center x coord
        self.cx = self.x0 + (self.w//2)
        # center y coord
        self.cy = self.y0 + (self.h//2)
        self.idx = idx
    
    def distance(self, other):
        assert(type(other) == Face)
        a = (self.cx, self.cy)
        b = (other.cx, other.cy)
        return dist(a, b)

    def get_hashables(self):
        return (self.x0, self.y0, self.x1, self.y1)
    
    def __hash__(self):
        return hash(self.get_hashables())

    def __equal__(self, other):
        if type(other) != Face:
            return False
        return self.get_hashables() == other.get_hashables()

def get_num_eyes(frame, f, eyes_cascade):
    faceROI = frame[f.y0:f.y1,f.x0:f.x1]
    #-- In each face, detect eyes
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2,y2,w2,h2) in eyes:
        eye_center = (f.x0 + x2 + w2//2, f.y0 + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        # frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    return len(eyes)

def start():
    faceCascade = cv2.CascadeClassifier(cascPath)
    eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,1280)
    video_capture.set(4,720)

    wait_time = 3 # sec
    wait_cycles = 16
    num_waited = wait_cycles
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("no video captured")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            # minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        num_faces = len(faces)
        print(faces, "\n")
        faces = [Face(faces[i], i) for i in range(num_faces)]

        # wait for key press for ~1 ms
        key = cv2.waitKey(WAIT)

        if key%256 == 27: # esc
            print("Escape... closing")
            break
        
        bad_ones, combos = combinations(faces)

        # # Draw a rectangle around the faces
        for i in range(num_faces):
            f = faces[i]
            top_left = (f.x0, f.y0)
            bottom_right = (f.x1, f.y1)

            num_eyes = get_num_eyes(frame, f, eyes_cascade)

            if num_eyes == 1:
                color=yellow
                if num_waited>wait_cycles:
                    num_waited = 0
                    audio = ["detect.mp3", "refrain.mp3"]
                    r = random.randint(0,len(audio)-1)
                    choice = audio[r]

                    def playy():
                        os.system( "afplay " + choice)
                    P = Process(name="",target=playy)
                    P.start() # Inititialize Process

            elif f in bad_ones:
                color=red
            else:
                color=green
            thickness=3 
            cv2.rectangle(frame, top_left, bottom_right, color, thickness)

        for (f1, f2) in combos:
            color =  red
            thickness = 3
            distance_in_pixels =f1.distance(f2)
            avg_distance= ( f1.w + f2.w )/2
            pixels_p_ft= (NUM_WIDTHS * avg_distance)/6
            distance_in_feet= distance_in_pixels / pixels_p_ft 

            if f1.x0 < f2.x0:
                start = (f1.x1, f1.cy)
                end = (f2.x0, f2.cy) #end on right side of box 2
                coord= ( int( (f1.x1 + f2.x0)/2), int( (f1.cy + f2.cy)/2))
            else:
                start = (f2.x1, f2.cy)
                end = (f1.x0 , f1.cy) # end on left side of box 2
                coord= ( int( (f2.x1 + f1.x0)/2), int( (f1.cy + f2.cy)/2))

            cv2.line(frame, start, end, color, thickness)
            s = "%.1f ft" % distance_in_feet
            coord = (coord[0]-10, coord[1]-15)
            cv2.putText(frame, s, coord, cv2.FONT_HERSHEY_SIMPLEX,  \
                   1, color, 2, cv2.LINE_AA) 

        if len(bad_ones) > 0 and num_waited >= wait_cycles:
            num_waited = 0
            audio = ["pleasekeep.mp3", "cdcrec.mp3"]
            r = random.randint(0,len(audio)-1)
            choice = audio[r]

            def playy():
                os.system( "afplay " + choice)
            P = Process(name="",target=playy)
            P.start() # Inititialize Process


        # Display the resulting frame
        cv2.imshow('Video', frame)
        num_waited+=1

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()