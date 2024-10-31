# other imports
import tobii_research as tr
# local imports
from .base_tracker import EyeTracker, TrackerData

class TobiiTracker(EyeTracker):

    name = 'Tobii Eye Tracker'

    def __init__(self):
        super().__init__()

        # try to find tobii trackers, then get first one
        try:
            found_eyetracker = tr.find_all_eyetrackers()
            self.eyetracker = found_eyetracker[0]
        except IndexError:
            pass

    def start_eye_tracker(self):
        if self.has_tracker():
            self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.tobii_gaze_data_callback, as_dictionary=True)

    def stop_eye_tracker(self):
        if self.has_tracker():
            self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.tobii_gaze_data_callback)

    def tobii_gaze_data_callback(self, gaze_data):
        data = TrackerData()
        
        data.timestamp = {
            'device': gaze_data['device_time_stamp'],
            'system': gaze_data['system_time_stamp']
        }
        data.left_eye = {
            'gaze_point_on_display_area': gaze_data['left_gaze_point_on_display_area'],
            'gaze_point_in_user_coordinate_system': gaze_data['left_gaze_point_in_user_coordinate_system'],
            'gaze_point_validity': gaze_data['left_gaze_point_validity'],
            'pupil_diameter': gaze_data['left_pupil_diameter'],
            'pupil_validity': gaze_data['left_pupil_validity'],
            'gaze_origin_in_user_coordinate_system': gaze_data['left_gaze_origin_in_user_coordinate_system'],
            'gaze_origin_in_trackbox_coordinate_system': gaze_data['left_gaze_origin_in_trackbox_coordinate_system'],
            'gaze_origin_validity': gaze_data['left_gaze_origin_validity']
        }
        data.right_eye = {
            'gaze_point_on_display_area': gaze_data['right_gaze_point_on_display_area'],
            'gaze_point_in_user_coordinate_system': gaze_data['right_gaze_point_in_user_coordinate_system'],
            'gaze_point_validity': gaze_data['right_gaze_point_validity'],
            'pupil_diameter': gaze_data['right_pupil_diameter'],
            'pupil_validity': gaze_data['right_pupil_validity'],
            'gaze_origin_in_user_coordinate_system': gaze_data['right_gaze_origin_in_user_coordinate_system'],
            'gaze_origin_in_trackbox_coordinate_system': gaze_data['right_gaze_origin_in_trackbox_coordinate_system'],
            'gaze_origin_validity': gaze_data['right_gaze_origin_validity']
        }
        self.gaze_data_callback(data)

tracker_class = TobiiTracker