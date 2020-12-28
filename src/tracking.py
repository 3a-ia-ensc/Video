# -*- coding: utf-8 -*-

""" tracking.py
This module is a specific module that allows tracking objects in video
"""

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020, Projet vidéo"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def read_bounding_box(path):
    """ Read the bouding box file of a video

    Parameters:
    path (string): the path for the query ressource

    Return:
    dict: the boxes parsed from the file
    """
    boxes = {}
    # m = re.search('VIDEOS/(.+)\.mp4', path)
    # filename = m.group(1)

    f = open(path, 'r')
    lines = f.readlines()

    for l in lines:
        elmts = l.split(' ')
        frame = int(elmts[0])

        if elmts[1] == '1':
            boxes[frame] = (int(elmts[2]), int(elmts[3]), int(elmts[4]), int(elmts[5]))

    return boxes


def build_boxes(x, y, w, h, N, stride, H, W):
    """ Build a set of new potential bouding boxes from a given box

    Parameters:
    x (int): the x coordinate
    y (int): the y coordinate
    w (int): width of the original box
    h (int): height of the original box
    N (int): determine the shift speed
    stride (int): determine the shift stride
    H (int): height of the frame (to manage out of bounds)
    W (int): width of the frame (to manage out of bounds)

    Return:
    list: list of new boxes
    """
    shift = list(range(-N, N+1, stride))
    zoom = [0.89, 0.76, 1, 1.1, 1.25]

    pos = [(x, y, w, h)]
    for i in shift:
        new_x = x + i

        for j in shift:
            new_y = y + j

            new_pos = (new_x, new_y, w, h)
            pos.append(new_pos)

            for z in zoom:
                new_w, new_h = int(w*z), int(h*z)


                new_x = new_x + ((w - new_w) // 2)
                new_y = new_y + ((h - new_h) // 2)

                new_x = min(max(0, new_x), W)
                new_y = min(max(0, new_y), H)

                new_w = min((W - new_x), new_w)
                new_h = min((H - new_y), new_h)

                pos.append((new_x, new_y, new_w, new_h))

    return pos


def tracking(model, video_path, box_path):
    """ Build a set of new potential bouding boxes from a given box

    Parameters:
    filename (string): path of the file to track

    Return:
    list: list of new boxes
    """
    labels = { 0: 'bol', 1: 'coca', 2: 'lait', 3: 'riz', 4: 'sucre' }
    # filename = '../VIDEOS/RicePlace3Subject3.mp4'

    m = re.search('VIDEOS/(.+)\.mp4', video_path)
    file = m.group(1)

    file_pred = open(f"{file}_pred_box.txt", "a")

    # on ouvre le fichier
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # on récupère les boundings box du fichier
    boxes = read_bounding_box(box_path)
    ious = {}
    predictions = {}

    if(not cap.isOpened()):
        print("ERROR: unable to read video:", video_filename)
        sys.exit()

    print(f'Loading {nframes} frames')

    if 'CanOfCocaCola' in video_path:
        label = 1
    elif 'MilkBottle' in video_path:
        label = 2
    elif 'Bowl' in video_path:
        label = 0
    elif 'Rice' in video_path:
        label = 3
    elif 'Sugar' in video_path:
        label = 4

    foundObj = False
    for f in tqdm(range(nframes)):
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f_width, f_height, _ = frame.shape

        if foundObj or f in boxes:
            if not foundObj:
                box = boxes[f] # première bounding box (= box de référence)
            else:
                x, y, w, h = box
                new_boxes = build_boxes(x, y, w, h, 100, 25, f_width, f_height)

                #fig,ax = plt.subplots(1)
                #plt.imshow(frame)

                pred_boxes = np.zeros((0, 227, 227, 3))

                t = 0

                for potential_box in new_boxes:
                    t+= 1
                    x1, y1, w1, h1 = potential_box

                    new_patch = frame[y1:y1+h1, x1:x1+w1, :]

                    if 0 not in new_patch.shape:
                        new_patch = tf.image.resize(new_patch, (227, 227), method='nearest')
                        pred_boxes = tf.concat((pred_boxes, tf.expand_dims(new_patch, 0)), axis=0)

                if pred_boxes.shape[0] > 0:
                    pred = model.predict(pred_boxes / 255)
                    best_pred = np.argmax(pred[:, label])

                    # on récupère le patch associé à la meilleure prédiction
                    best_box = new_boxes[best_pred]
                    x2, y2, w2, h2 = best_box
                    #rect = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='y', facecolor='none')
                    #ax.add_patch(rect)

                    # on modifie la current box pour la frame suivante
                    box = best_box
                    predictions[f] = box
                    file_pred.write(f"{f} {1} {box[0]} {box[1]} {box[2]} {box[3]}\n")

                if f in boxes:
                    a, b, c, d = boxes[f]
                    #rect = patches.Rectangle((a, b), c, d, linewidth=1, edgecolor='g', facecolor='none')
                    #ax.add_patch(rect)

                    iou = bb_intersection_over_union(box, boxes[f])
                    ious[f] = iou

                    # plt.savefig(f'img/frame__best{f}.png')

            foundObj = True

    return predictions

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    a = boxA[0] + boxA[2]
    b = boxA[1] + boxA[3]

    c = boxB[0] + boxB[2]
    d = boxB[1] + boxB[3]

    xB = min(a, c)
    yB = min(b, d)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
