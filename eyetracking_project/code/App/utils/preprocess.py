import pandas as pd
import numpy as np
import math

# CONSTANTS
# Display Resolution
MAXIMAL_DISTANCE = 25 # Eucledian Distance Fixation Calculation (Dispersion)
MINIMAL_DURATION = 50 # Minimal duration to be considered a fixation in milliseconds (typically 50 - 200)

def prepare_raw_data(gazeData, dataset, dispaly_size):
    # preprocess the dataframe , remove noise
    if gazeData.empty:
        print("Warning: gazeData is empty. Skipping processing.")
        return

    gazeData = compute_mean_gaze_point(gazeData)
    gazeData = filterOffDisplayValues(gazeData)
    timestamps = getTimestampsInMilliseconds(gazeData)
    time = pd.Series(timestamps)
    gazeData = gazeData.assign(Timestamp=time.values)
    gazeData.reset_index()

    x_gazePoints, width = getXGazePointsAsPixel(gazeData, dispaly_size)
    y_gazePoints, height = getYGazePointsAsPixel(gazeData, dispaly_size)
    s = pd.Series(x_gazePoints)
    t = pd.Series(y_gazePoints)
    gazeData = gazeData.assign(PixelPointX=s.values)
    gazeData = gazeData.assign(PixelPointY=t.values)
    gazeData = remove_noise_df(gazeData)
    gazeData = calculate_isd(gazeData)
    gazeData = add_velocity_df(gazeData)

    gazeData.to_csv(dataset, index=False, na_rep='NaN')


def event_detecting_smoothing(csv_file):
    event_df = pd.read_csv(csv_file)
    if event_df.empty:
        print("Warning: event_df is empty. Skipping processing.")
        return
    else:
        gazeData = mark_events_saccade_blink_noise(csv_file)
        gazeData = smooth_gaze_data(gazeData)
        gazeData.to_csv(csv_file, index=False, na_rep='NaN')

def get_fixations(csv_file):
    # get the fixations
    gazeData = pd.read_csv(csv_file)
    timestamps = getTimestampsInMilliseconds(gazeData)
    x_gazePoints = gazeData.SmoothPixelPointX
    y_gazePoints = gazeData.SmoothPixelPointY
    fixations_all_data = FilterDataWithFixations_weighted_mean(x_gazePoints, y_gazePoints, timestamps, mindur=50)
    fixations = []
    for fixation_data in fixations_all_data:
        start_time, end_time, duration, mean_x, mean_y, _ = fixation_data  # Ignore fixation points
        fixations.append([start_time, end_time, duration, mean_x, mean_y])
    return fixations



''' Fixation Algorithm'''

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def calculate_weighted_mean(points):
    # Calculate the weighted mean of a list of points
    n = len(points)
    weighted_x_sum = sum((i + 1) * point.x for i, point in enumerate(points))
    weighted_y_sum = sum((i + 1) * point.y for i, point in enumerate(points))
    total_weight = (n * (n + 1)) / 2  # Sum of weights from 1 to n

    if total_weight == 0:
        return None  # Avoid division by zero

    mean_x = weighted_x_sum / total_weight
    mean_y = weighted_y_sum / total_weight

    return Point(mean_x, mean_y)

def FilterDataWithFixations_weighted_mean(x, y, time, saccade_threshold=MAXIMAL_DISTANCE, maxdist=MAXIMAL_DISTANCE, mindur=MINIMAL_DURATION):
    fixations = []

    current_fixation = []

    for i in range(len(x)):
        current_point = Point(x[i], y[i])

        if not current_fixation:
            current_fixation.append(current_point)
            continue

        last_fixation_point = current_fixation[-1]
        distance_to_current = calculate_distance(current_point, last_fixation_point)

        if distance_to_current < saccade_threshold:
            current_fixation.append(current_point)
        else:
            if current_fixation:
                # Check if the current fixation is valid (meets duration criteria)
                start_time = time[i - len(current_fixation)]
                end_time = time[i - 1]
                duration = end_time - start_time
                if duration >= mindur:
                    weighted_mean = calculate_weighted_mean(current_fixation)
                    if weighted_mean is not None:
                        fixations.append([start_time, end_time, duration, weighted_mean.x, weighted_mean.y, current_fixation])
                current_fixation = []

    # Check if there's any remaining fixation
    if current_fixation:
        start_time = time[len(x) - len(current_fixation)]
        end_time = time[-1]
        duration = end_time - start_time
        if duration >= mindur:
            weighted_mean = calculate_weighted_mean(current_fixation)
            if weighted_mean is not None:
                fixations.append([start_time, end_time, duration, weighted_mean.x, weighted_mean.y, current_fixation])

    return fixations


''' Helper Functions'''


def compute_mean_gaze_point(dataframe):
    actualGazePointX = []
    actualGazePointY = []

    for index, row in dataframe.iterrows():
        leftX = row['left_gaze_point_x']
        leftY = 1-row['left_gaze_point_y']
        rightX = row['right_gaze_point_x']
        rightY = 1-row['right_gaze_point_y']

        if not np.isnan(leftX) and not np.isnan(leftY) and not np.isnan(rightX) and not np.isnan(rightY):
            actualX = (leftX + rightX) / 2
            actualY = (leftY + rightY) / 2
        elif np.isnan(leftX) and np.isnan(leftY):
            actualX = rightX
            actualY = rightY
        elif np.isnan(rightX) and np.isnan(rightY):
            actualX = leftX
            actualY = leftY

        actualGazePointX.append(actualX)
        actualGazePointY.append(actualY)

    s = pd.Series(actualGazePointX)
    t = pd.Series(actualGazePointY)
    dataframe = dataframe.assign(ActualGazePointX=s.values)
    dataframe = dataframe.assign(ActualGazePointY=t.values)
    return dataframe


#Remove all off Display values
def filterOffDisplayValues(dataframe):

    inboundAll = dataframe[dataframe['ActualGazePointX'] <= 1]
    inboundAll = inboundAll[inboundAll['ActualGazePointX'] >= 0]

    inboundAll = inboundAll[inboundAll['ActualGazePointY'] <= 1]
    inboundAll = inboundAll[inboundAll['ActualGazePointY'] >= 0]

    return inboundAll


def getTimestampsInMilliseconds(dataframe):
    time = dataframe["timestamp_system"]
    # time = dataframe.Timestamp
    time.reset_index(drop=True, inplace=True)
    t_temp = []

    if not time.empty:
        initalTime = time.iloc[0] / 1000
        for t in time:
            t_temp.append(t / 1000 - initalTime)
    else:
        print("Warning: time Series is empty. Returning an empty list.")

    return t_temp

def getXGazePointsAsPixel(dataframe, dispaly_size):
    # user32 = ctypes.windll.user32
    # width = user32.GetSystemMetrics(0)
    # print(width)
    width = dispaly_size[0]
    # width =  2560 #2048 #412
    x_temp = []
    actualX = dataframe['ActualGazePointX']
    for x in actualX[0:]:
        x_temp.append(round(x*width))
        # x_temp.append(x * width)
    return x_temp, width


def getYGazePointsAsPixel(dataframe, dispaly_size):
    # user32 = ctypes.windll.user32
    # height = user32.GetSystemMetrics(1)
    # print(height)
    height = dispaly_size[1]
    # height = 1440 #1152 #1440 #915
    y_temp = []
    actualY = dataframe['ActualGazePointY']
    for y in actualY[0:]:
        y_temp.append(round(y*height))
        # y_temp.append(y * height)
    return y_temp, height


def calculate_isd(df):
    df['InterSampleDuration_DS'] = df["timestamp_system"].diff().fillna(0) / 1000
    return df

def remove_noise_df(dataframe):
    gazePointX = dataframe['PixelPointX'].tolist()
    gazePointY = dataframe['PixelPointY'].tolist()
    # threshold = 0.97 degrees X; 1.17 degrees Y on a 2560x1440 disp at 60cm-65cm distance
    noNoiseGazePointX = replace_outliers_with_median(gazePointX, threshold=50)
    noNoiseGazePointY = replace_outliers_with_median(gazePointY, threshold=60)

    dataframe['PixelPointX'] = noNoiseGazePointX
    dataframe['PixelPointY'] = noNoiseGazePointY

    return dataframe

def replace_outliers_with_median(points, threshold, window_size=3):
    num_points = len(points)
    new_points = []

    for i in range(len(points)):
        if i < window_size or i >= num_points - window_size:
            # not enough points
            new_points.append(points[i])
        else:
            # Euclidean to points within the +2 and -2 window
            distances = []
            for j in range(i - window_size, i + window_size + 1):
                distance = np.linalg.norm(points[i] - points[j])
                distances.append(distance)

            # distance greater than the threshold -->  outlier
            if np.mean(distances) > threshold:
                # Replace the outlier point with the median of the points within the window
                median_point = np.median([points[j] for j in range(i - window_size, i + window_size + 1) if j != i],
                                         axis=0)
                new_points.append(median_point)

            else:
                new_points.append(points[i])

    return new_points

def add_velocity_df(gazeData, sampling_frequency=250):
    # timestamps = gazeData.Timestamp
    timestamps = getTimestampsInMilliseconds(gazeData)
    x_gazePoints = gazeData.PixelPointX
    y_gazePoints = gazeData.PixelPointY
    velocity = calculate_velocity(x_gazePoints, y_gazePoints, timestamps, sampling_frequency)
    # print(len(velocity), len(gazeData))
    if len(velocity) == len(gazeData) - 1:
        velocity = np.insert(velocity, 0, 0)
    elif len(velocity) == len(gazeData):
        velocity = velocity
        # s = pd.Series(velocity)
        # gazeData = gazeData.assign(Velocity=s.values)
    gazeData['Velocity'] = velocity

    return gazeData

def calculate_velocity(x, y, timestamps, sampling_frequency=250):
    time_intervals = np.diff(timestamps) / 1000  # Convert milliseconds to seconds

    if np.any(time_intervals == 0):
        print("Warning: Velocity calculation forced.")
        # force replace zero time intervals with 4ms because ET error, should be changed for different sampliong
        time_intervals[time_intervals == 0] = 4

    velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / (time_intervals * sampling_frequency)
    velocity = np.insert(velocity, 0, 0)

    return velocity


def mark_events_saccade_blink_noise(csv_file):
    event_df = pd.read_csv(csv_file)
    if event_df.empty:
        print("Warning: gazeData is empty. Skipping processing.")
        return

    else:
        event_df['Event_Type'] = 'Unknown'

        saccade_start = False
        saccade_end = False

        for i in range(len(event_df)):
            if i < 0:
                continue

            inter_sample_duration = event_df.at[i, 'InterSampleDuration_DS']
            velocity = event_df.at[i, 'Velocity']

            if 99 < inter_sample_duration < 200:
                event_df.at[i, 'Event_Type'] = 'Blink_End'
            elif inter_sample_duration > 200:
                event_df.at[i, 'Event_Type'] = 'Noise_End'
            else:
                if velocity > 30 and not saccade_start:
                    event_df.at[i, 'Event_Type'] = 'Saccade_Start'
                    saccade_start = True
                    saccade_end = False

                    # Check if the next row exists
                    if i + 1 < len(event_df):
                        i = i + 1

                if saccade_start and i + 1 < len(event_df) and event_df.at[i + 1, 'Velocity'] <= 30 and not saccade_end:
                    event_df.at[i-1, 'Event_Type'] = 'Saccade_End'
                    saccade_start = False
                    saccade_end = True

                if saccade_start and not saccade_end:
                    event_df.at[i, 'Event_Type'] = 'Saccade'

        for i in range(len(event_df)):
            if i == 0:
                continue
            if event_df.at[i, 'Event_Type'] == 'Saccade_End' and event_df.at[i - 1, 'Event_Type'] == 'Saccade_Start':
                event_df.at[i, 'Event_Type'] = 'Noise'
                event_df.at[i - 1, 'Event_Type'] = 'Noise'
            if event_df.at[i, 'Event_Type'] == 'Saccade_End' and event_df.at[i - 1, 'Event_Type'] != 'Saccade':
                event_df.at[i, 'Event_Type'] = 'Noise'

        return event_df



def smooth_gaze_data(dataframe):
    if dataframe.empty:
        print("Warning: gazeData is empty. Skipping processing.")
        return

    gazePointX = dataframe['PixelPointX'].tolist()
    gazePointY = dataframe['PixelPointY'].tolist()
    eventType = dataframe['Event_Type']

    window_start = 0
    i = 0

    while i < len(dataframe):
        if eventType[i] == 'Saccade_Start':
            window_end = i - 1
            if window_start <= window_end:

                noNoiseGazePointX = apply_filter(gazePointX[window_start:window_end + 1])
                noNoiseGazePointY = apply_filter(gazePointY[window_start:window_end + 1])

                j = i + 1
                while j < len(dataframe) and eventType[j] != 'Saccade_End':
                    j += 1

                gazePointX[window_start:window_end+1] = noNoiseGazePointX
                gazePointY[window_start:window_end+1] = noNoiseGazePointY

                window_start = j - 1 # Early Onset of Filtering to avoid problem stated at func start

            i = window_start
        else:
            i += 1

    if window_start < len(dataframe):
        noNoiseGazePointX = apply_filter(gazePointX[window_start:])
        noNoiseGazePointY = apply_filter(gazePointY[window_start:])
        gazePointX[window_start:] = noNoiseGazePointX
        gazePointY[window_start:] = noNoiseGazePointY

    dataframe['SmoothPixelPointX'] = gazePointX
    dataframe['SmoothPixelPointY'] = gazePointY

    return dataframe


def apply_filter(gazePoints):
    window_length = 4
    filteredPoints = []
    gaussian_weight_func = gaussian_filter(window_length)
    i = 0  # Set i = 0 before while loop

    while i < len(gazePoints):
        if i < window_length:
            # window_data = gazePoints[i:i + window_length]
            # filteredPoints.append(np.median(window_data, axis=0))
            filteredPoints.append(gazePoints[i])
        else:
            temp_points = gazePoints[(i - window_length + 1):(i + 1)]
            weighted_avg = np.average(a=temp_points, weights=gaussian_weight_func, axis=0)
            filteredPoints.append(weighted_avg)
        i = i + 1  # Increment i within loop
    return filteredPoints


def gaussian_filter(window_length):
    gaze_window_list = np.arange(1, window_length + 1)
    gauss_list = []
    for x in gaze_window_list:
        x = math.e ** -((x ** 2)/(2 * (65)))
        gauss_list.append(x)
    gauss_list.reverse()
    return gauss_list
