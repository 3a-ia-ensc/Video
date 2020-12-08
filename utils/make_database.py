import os
import sys
import numpy as np
import glob
import cv2


def usage(prog):
    print("Usage: {} video_dir bboxes_dir output_dir".format(prog))
    sys.exit()

num_args=len(sys.argv)
if (num_args != 4):
    usage(sys.argv[0])
video_dir = sys.argv[1]
bboxes_dir = sys.argv[2]
output_dir = sys.argv[3]

need_squares = True
patch_width = 227
patch_height = 227



def extract_patches(video_filename, bboxes_filename, output_dir, show=False):

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

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        if show:
            disp = frame.copy()
    
        if frame_number in bboxes:
            x, y, w, h = bboxes[frame_number]

            if (x<0):
                x=0
            if (y<0):
                y=0
            if (x+w>frame_width):
                w=frame_width-x
            if (y+h>frame_height):
                h=frame_height-y

            if show:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 255), 2) 
        
            rects = []
        
            if need_squares:
                #first, try to extend rectangle to square
                if w < h:
                    w2 = h
                    x2 = int(x+w/2-w2/2)
                    y2 = y
                    h2 = h
                else:
                    h2 = w
                    y2 = int(y+h/2-h2/2)
                    x2 = x
                    w2 = w
                
                if not (x2>=0 and y2>=0 and x2+w2<=frame_width and y2+h2<=frame_height):
                    #keep biggest possible square (centered on original bbox)
                    xc = x+w/2
                    yc = y+h/2
                    xl = max(0, xc-w2/2)
                    xr = min(xc+w2/2, frame_width)
                    w2 = 2 * int(min(xc-xl, xr-xc))
                    x2 = int(xc-w2/2)
                    
                    yu = max(0, yc-h2/2)
                    yd = min(yc+h2/2, frame_height)
                    h2 = 2 * int(min(yc-yu, yd-yc))
                    y2 = int(yc-h2/2)

                assert(x2>=0)
                assert(y2>=0)
                assert(x2+w2<=frame_width)
                assert(y2+h2<=frame_height)
                rects.append((x2, y2, w2, h2))
                
            for i in range(len(rects)):
                x, y, w, h = rects[i]
            
                patch = frame[y:y+h, x:x+w]
                resized = cv2.resize(patch, (patch_width, patch_height), interpolation = cv2.INTER_LINEAR) 
                
                output_filename=os.path.join(output_dir, os.path.basename(bboxes_filename).replace(".txt", "")+"_{}_{}.png".format(frame_number, i))
                #print("output_filename=", output_filename)
                cv2.imwrite(output_filename, resized)

                if show:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2) 

        if show:
            cv2.imshow('frame', disp)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        frame_number += 1
        
    cap.release()



ratio_train=0.8


train_dir=os.path.join(output_dir, "train")
test_dir=os.path.join(output_dir, "test")
os.mkdir(train_dir)
os.mkdir(test_dir)

for category in ["Bowl", "CanOfCocaCola", "MilkBottle", "Rice", "Sugar"]:
    bboxes_files = sorted(glob.glob(os.path.join(bboxes_dir, category+"*_2_bboxes.txt")))

    #Some objects are present in just one place.
    #We want to divide dataset in train & test, such as no place is in test and not in train.

    category_train_dir = os.path.join(train_dir, category)
    category_test_dir = os.path.join(test_dir, category)
    
    os.mkdir(category_train_dir)
    os.mkdir(category_test_dir)
    
    
    train_files = []
    test_files = []
    
    h_places={}
    for bboxes_file in bboxes_files:
        bb_file = os.path.basename(bboxes_file)
        prefix=category+"Place"
        assert(bb_file.startswith(prefix))
        assert(len(bb_file) > len(prefix))
        place=bb_file[len(prefix)]
        if place in h_places:
            h_places[place].append(bb_file)
        else:
            h_places[place] = [bb_file]
    for p in h_places:
        files = h_places[p]
        num=len(files)
        if num == 1:
            train_files.append(os.path.join(bboxes_dir, files[0]))
        else:
            assert(num > 1)
            num_train = (int)(num*ratio_train)
            for i in range(num_train):
                train_files.append(os.path.join(bboxes_dir, files[i]))
            for i in range(num_train,num):
                test_files.append(os.path.join(bboxes_dir, files[i]))

    for bbox_file in train_files:
        video_filename = os.path.join(video_dir, os.path.basename(bbox_file).replace("_2_bboxes.txt", ".mp4"))
        if not os.path.exists(video_filename):
            print("ERROR: video file not found:", video_filename)
            sys.exit()
        extract_patches(video_filename, bbox_file, category_train_dir)

    for bbox_file in test_files:
        video_filename = os.path.join(video_dir, os.path.basename(bbox_file).replace("_2_bboxes.txt", ".mp4"))
        if not os.path.exists(video_filename):
            print("ERROR: video file not found:", video_filename)
            sys.exit()
        extract_patches(video_filename, bbox_file, category_test_dir)

        
