import os
import sys
import numpy as np
import cv2


def usage(prog):
    print("Usage: {} video_filename bboxes_filename".format(prog))
    sys.exit()

num_args=len(sys.argv)
if (num_args != 3):
    usage(sys.argv[0])
video_filename = sys.argv[1]
bboxes_filename = sys.argv[2]


cap = cv2.VideoCapture(video_filename)
if (not cap.isOpened()):
    print("ERROR: unable to read video:", video_filename)
    sys.exit()


bboxes = {}
with open(bboxes_filename) as fp: 
    lines = fp.readlines() 
    for line in lines: 
        elts = line.split()
        if len(elts) == 2:
            assert(elts[1] == "0")
        else:
            assert(len(elts) > 2)
            assert(elts[1] == "1") #only one bbox per frame
            frame_number = (int)(elts[0])
            x = (int)(elts[2])
            y = (int)(elts[3])
            w = (int)(elts[4])
            h = (int)(elts[5])
            bboxes[frame_number] = (x, y, w, h)

     
frame_number=1 #start from 1 in bboxes_filename

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        break
    
    if frame_number in bboxes:
        x, y, w, h = bboxes[frame_number]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    frame_number += 1
    
cap.release()
