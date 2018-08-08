# created by Sijmen van der Willik
# 05/08/2018 15:34
#
# predicts location of vehicles in an image

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import find_cars


previous_boxes = []

count = 0


def predict_vehicle(img, n_steps=10, threshold=20, mine_boxes=False):
    """Draws vehicle detection on input images


    Requires global boxes_history <list of lists of lists of tuples or lists> boxes detected in earlier time steps
        i.e. two time steps with 2 predicted boxes each
        [
            [ [[100, 100], [300, 300]], [[400, 200], [500, 300]] ],
            [ [[120, 120], [320, 320]], [[420, 220], [520, 320]] ]
        ]

    :param img: <numpy array> the image the be used
    :param n_steps: <int> number of time steps to go back for checking boxes
    :param threshold: <int> min number of detections to be considered True
    :param mine_boxes: <bool> if True, saves patches of any detected boxes
    :return: <list of list of tuples> List of coordinate sets
    """
    global previous_boxes
    global count

    previous_boxes.append(find_cars.get_all_boxes(img))

    heat_map = np.zeros(img.shape[:2])

    # if mine_boxes is True, save the detected box as a separate image so it can be used in training later
    if mine_boxes:
        for box in previous_boxes[-1]:
            count += 1
            patch = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            mpimg.imsave("./images/mining/{}_{:0>5}.png".format(box[0][0], count), patch)

    # build heat map from boxes
    for time_step in previous_boxes[-n_steps:]:
        for box in time_step:
            heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Zero out pixels below the threshold
    heat_map[heat_map < threshold] = 0

    # change img
    img[heat_map > 0] = np.array([255, 0, 0])

    return img
