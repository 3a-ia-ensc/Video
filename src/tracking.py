# -*- coding: utf-8 -*-

""" tracking.py
This module is a specific module that allows tracking objects in video
"""

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020, Projet vidéo"
__license__ = "WTFPL"
__version__ = "2.0.1"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"

import re
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as plt_patch
from tqdm import tqdm
import os


def get_label(video_path):
    """ Get the label from a video filename

    Args:
        video_path (string): video path

    Returns: label (int)
    """
    m = re.search('VIDEOS/(.+)\.mp4', video_path)
    file = m.group(1)

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

    return label


def read_bounding_box(path):
    """ Read a bouding box file

    Parameters:
    path (string): the path for the query ressource

    Return:
    dict: the boxes parsed from the file
    """
    boxes = {}

    f = open(path, 'r')
    lines = f.readlines()

    for l in lines:
        elmts = l.split(' ')
        frame = int(elmts[0])

        if elmts[1] == '1':
            boxes[frame] = (int(elmts[2]), int(elmts[3]), int(elmts[4]), int(elmts[5]))

    return boxes


def build_boxes(box, shiftSize, stride, zoom, H, W):
    """ Build a set of new potential bouding boxes from a given box

    Args:
        box (tuple): box data (format: (x, y, width, height))
        shiftSize (int): used to compute the box shift from original position
        stride (int): determine the shift stride, step to use in shifting computation
        zoom (list): zoom factors
        H (int): height of the frame (to manage out of bounds)
        W (int): width of the frame (to manage out of bounds)

    Returns:
    list: list of new boxes
    """
    x, y, w, h = box
    shift = list(range(-shiftSize, shiftSize + 2, stride))

    res = [(x, y, w, h)]
    # shift in two directions
    for i in shift:
        new_x = x + i

        for j in shift:
            new_y = y + j

            new_pos = (new_x, new_y, w, h)
            res.append(new_pos)

    # zoom every new box
    N = len(res)
    zooms = []
    for i in range(N):
        a, b, c, d = res[i]

        for z in zoom:
            new_w, new_h = int(c * z), int(d * z)

            new_x = a + ((c - new_w) // 2)
            new_y = b + ((d - new_h) // 2)

            new_x = min(max(0, new_x), W)
            new_y = min(max(0, new_y), H)

            new_w = min((W - new_x), new_w)
            new_h = min((H - new_y), new_h)

            res.append((new_x, new_y, new_w, new_h))
            zooms.append((new_x, new_y, new_w, new_h))

    return np.array(res)


def tracking_image(frame, box, model, label, f_width, f_height):
    """

    Args:
        frame (np.array): image to track
        box (tuple): original box (format: (x, y, w, h))
        model (tf.keras.Model): model to use
        label (int): label of the object to track
        f_width:
        f_height:

    Returns:

    """
    patch_size = 227
    zoom = [0.9, 1.1]
    new_boxes = build_boxes(box, 40, 20, zoom, f_width, f_height)
    best_box = None

    patches = np.empty((0, patch_size, patch_size, 3))

    for potential_box in new_boxes:
        x1, y1, w1, h1 = potential_box

        new_patch = frame[y1:y1 + h1, x1:x1 + w1, :]
        if 0 not in new_patch.shape:
            new_patch = tf.image.resize(new_patch, (patch_size, patch_size), method='nearest')
            patches = np.concatenate((patches, tf.expand_dims(new_patch, axis=0)), axis=0)

    if len(patches) > 0:
        prediction = model.predict(patches)
        best_prediction = np.argmax(prediction[:, label])

        best_box = new_boxes[best_prediction]

        return best_box, patches[best_prediction]
    else:
        x, y, w, h = box
        return box, frame[y:y + h, x:x + w, :]


def tracking(model, video_path, box_path, label, update=None, n_update=5, save_fig=False, epsilon=0.0001, name=None):
    """ Compute object tracking on every frame of a video file

    Args:
        model (tf.keras.Model): model to use
        video_path (string): path of the video file
        box_path (string): path of the bounding box file
        label (int): label of the object in the video
        update (string): enum (=move-to-data, fine-tuning, or None), determines if continuous learning should be use
        n_update (int): batch size to use in continuous learning
        save_fig (bool): if True, will save each frame with predicted box
        epsilon (float): learning rate for continuous learning

    Returns:
        predictions (dict): box prediction for each frame
        ious (dict): IoU computed on each frame that had an original box

    """
    # initialize file to save prediction
    if update == 'move-to-data':
        dir = 'move_to_data'
    elif update == 'fine-tuning':
        dir = 'fine_tuning'
    else:
        dir = 'normal'

    if name is None:
        filename = os.path.basename(box_path)
    else:
        filename = f'{name}_{os.path.basename(box_path)}'
    if os.path.exists(f"../GT_pred/{dir}/{filename}"):
        os.remove(f"../GT_pred/{dir}/{filename}")
    file_pred = open(f"../GT_pred/{dir}/{filename}", "a")

    # on ouvre le fichier
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Loading {nframes} frames')

    # on récupère les boundings box du fichier
    boxes = read_bounding_box(box_path)

    ious = {}
    predictions = {}
    update_batch = []

    foundObj = False
    for f in tqdm(range(nframes)):
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f_width, f_height, _ = frame.shape

        if foundObj or f in boxes:
            if not foundObj:
                box = boxes[f]  # première bounding box (= box de référence)
            else:
                box, patch = tracking_image(frame, box, model, label, f_width, f_height)
                predictions[f] = box
                file_pred.write(f"{f} {1} {box[0]} {box[1]} {box[2]} {box[3]}\n")

                if update == 'move-to-data':
                    if len(update_batch) < n_update:
                        update_batch.append(patch)
                    else:
                        move_to_data(model, np.array(update_batch), epsilon, label)
                        update_batch = []
                elif update == 'fine-tuning':
                    if len(update_batch) < n_update:
                        update_batch.append(patch)
                    else:
                        fine_tuning(model, np.array(update_batch), epsilon, label)
                        update_batch = []

                if f in boxes:
                    iou = intersection_over_union(box, boxes[f])
                    ious[f] = iou

                if save_fig:
                    x, y, w, h = box
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    plt.axis('off')
                    ax.imshow(frame)

                    # add the box on the image
                    rect = plt_patch.Rectangle((x, y), w, h, linewidth=1, edgecolor='y', facecolor='none')
                    ax.add_patch(rect)

                    if f in boxes:
                        # add real box
                        a, b, c, d = boxes[f]
                        rect = plt_patch.Rectangle((a, b), c, d, linewidth=1, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)

                    # save image
                    fig.savefig(f'../img_tracking/{dir}/{filename}_{f}.png')
                    plt.close(fig)

            foundObj = True

    return predictions, ious


def intersection_over_union(boxA, boxB):
    """ Computes intersection over union between box A and box B

    Args:
        boxA: a bounding box (format: (x, y, w, h))
        boxB: a bounding box (format: (x, y, w, h))

    Returns: iou (float)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    a = boxA[0] + boxA[2]
    b = boxA[1] + boxA[3]

    c = boxB[0] + boxB[2]
    d = boxB[1] + boxB[3]

    xB = min(a, c)
    yB = min(b, d)

    # intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # union
    unionArea = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea

    iou = abs(interArea) / abs(unionArea)
    return iou


def move_to_data(model, images, epsilon, j):
    """ Update method for continuous learning

    Args:
        model (tf.keras.Model): the model to fine tuning
        images (np.array): batch of images
        j (int): label of the training class
        epsilon (float): learning rate
    """
    outputs = [layer for layer in model.layers]
    last_layer = outputs[-1]
    w, b = last_layer.get_weights()
    new_weights = w.copy()

    # slight model modification to output last two layers
    mModel = tf.keras.Model(outputs[0].output, [outputs[-2].output, outputs[-1].output])
    mModel.compile(optimizer='adam', loss='categorical_crossentropy')
    y_, _ = mModel(images)

    w_j = w[:, j]
    new_w = w_j + (tf.norm(w_j) * (y_ / tf.norm(y_)) - w_j) * epsilon
    new_weights[:, j] = tf.reduce_mean(new_w, axis=0)

    last_layer.set_weights([new_weights, b])

def fine_tuning(model, images, epsilon, label):
    """ Update method for continuous learning

    Args:
        model (tf.keras.Model): the model to fine tuning
        images (np.array): batch of images
        label (int): label of the training class
        epsilon (float): learning rate
    """
    opt = tf.keras.optimizers.Adam(learning_rate=epsilon)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    for layer in model.layers[:-1]:
        layer.trainable = False

    if len(images) > 0 and 0 not in images.shape:
        labels = np.zeros([len(images), 5])
        labels[:, label] = 1
        model.fit(x=images, y=labels, epochs=1)


def tracker_open_cvmodel(video_path, box_path, tracker):
    """ Perform tracking using OpenCV tracker

    Args:
        video_path (string): path of the video file
        box_path (string): path of the boxes file
        tracker (string): name of the tracker to use

    Returns: mean IoU over each frame
    """
    trackers = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mil": cv2.TrackerMIL_create
    }

    tracker = trackers[tracker]()

    # on ouvre le fichier
    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Loading {nframes} frames')

    # on récupère les boundings box du fichier
    boxes = read_bounding_box(box_path)

    foundObj = False
    ious = []

    for f in tqdm(range(nframes)):
        ret, frame = cap.read()

        if foundObj or f in boxes:
            if not foundObj:
                box = boxes[f]  # première bounding box (= box de référence)

                tracker.init(frame, box)
            else:
                _, new_box = tracker.update(frame)
                # x1, y1, w1, h1 = new_box

                if f in boxes:
                    iou = intersection_over_union(new_box, boxes[f])
                    ious.append(iou)

            foundObj = True

    return np.mean(ious)
