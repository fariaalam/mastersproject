import pandas as pd
import numpy as np
from . import preprocess as preprocess
import math


# Function to get fixations from gaze data
def get_fixations_from_gaze(log_path, display_size):
    gaze_df = pd.read_csv(log_path, delimiter = ",", header=None, 
            names = ['timestamp_system','left_gaze_point_x', 'left_gaze_point_y',
                    "right_gaze_point_x", "right_gaze_point_y"])
    gaze_df.dropna(inplace=True)

    proccessed_csv_file_path = 'proccessed_gaze_data.csv'

    # preporcess the raw data
    preprocess.prepare_raw_data(gaze_df, proccessed_csv_file_path, display_size)
    # smooth the preprocessed data
    preprocess.event_detecting_smoothing(proccessed_csv_file_path)
    # get the fixations
    fixations = preprocess.get_fixations(proccessed_csv_file_path)
    return fixations

def calculate_saccades(fixations):
    # calculate saccades based distance based velocity
    saccades = []
    for i in range(1, len(fixations)):
        start_time_p, end_time_p, _ , fix_x_p, fix_y_p = fixations[i-1]
        start_time_c, end_time_c, _ , fix_x_c, fix_y_c = fixations[i] 

        dx = fix_x_c - fix_x_p
        dy = fix_y_c - fix_y_p
        duration = start_time_c - end_time_p
        velocity = math.sqrt(dx**2 + dy**2) / duration

        saccades.append((end_time_p, start_time_c, duration, dx, dy, velocity, 
                        fix_x_p, fix_y_p, fix_x_c, fix_y_c))
    
    # extract the velocities
    velocity = [element[5] for element in saccades]
    # calculate the threshold
    velocity_threshold = get_threshold(velocity)
    # based on the threshold calculate the saccades
    selected_saccades = [element for element in saccades if element[5] > velocity_threshold]
    return selected_saccades

def get_threshold(elements):
    # calculate the threshold using standard deviation and mean
    std = np.std(elements)
    avg = np.average(elements)
    threshold = avg + std
    return threshold

def calculate_forward_saccades(saccades):
    # calculate the forward saccades
    forward_saccades = []
    for saccade in saccades: 
        _, _, _, dx, dy, _, _, _, _, _ = saccade
        if dy == 0:
            if dx > 0:
                forward_saccades.append(saccade) 
        elif dy > 0:
            forward_saccades.append(saccade)

    return forward_saccades


def calculate_backward_saccades(saccades):
    # calculate the backward saccades
    backward_saccades = []
    for saccade in saccades: 
        _, _, _, dx, dy, _, _, _, _, _ = saccade
        if dy == 0:
            if dx < 0:
               backward_saccades.append(saccade)
        elif dy < 0:
           backward_saccades.append(saccade)

    return backward_saccades

def aoi_saccades_counts(aois, saccades):
    # calculate the number of saccades in each aoi
    saccades_counts = []
    for index, aoi in enumerate(aois.items()):
        key, value = aoi
        aoi_x, aoi_y, aoi_w, aoi_h = value
        counts = 0
        for element in saccades:
            _, _, _, dx, dy, _, x1, y1, x2, y2 = element
            if aoi_x - 20 <= x1 <= aoi_x + aoi_w + 20 and aoi_y - 20 <= y1 <= aoi_y + aoi_h + 20 and \
            aoi_x - 20 <= x2 <= aoi_x + aoi_w + 20 and aoi_y - 20 <= y2 <= aoi_y + aoi_h + 20 :
                counts = counts + 1
        saccades_counts.append(counts)
    return saccades_counts


def calculate_saccades_length_AOIs(aois, saccades):
    # calculate the length of the saccades present in each aois
    saccades_length = {}
    for index, aoi in enumerate(aois.items()):
        saccades_length[index] = []
    for saccade in saccades:
        _, _, _, dx, dy, _, x1, y1, x2, y2 = saccade
        length = math.sqrt(dx**2 + dy**2)
        for index, aoi in enumerate(aois.items()):
            key, value = aoi
            aoi_x, aoi_y, aoi_w, aoi_h = value
            if aoi_x - 20 <= x1 <= aoi_x + aoi_w + 20 and aoi_y - 20 <= y1 <= aoi_y + aoi_h + 20 and \
            aoi_x - 20 <= x2 <= aoi_x + aoi_w + 20 and aoi_y - 20 <= y2 <= aoi_y + aoi_h + 20 :
                saccades_length[index].append(length)
    return saccades_length



