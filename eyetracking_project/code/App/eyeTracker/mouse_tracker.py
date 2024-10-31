# kivy imports
from kivy.clock import Clock
# other imports
from time import time
import screeninfo
import pyautogui
# local imports
from .base_tracker import EyeTracker, TrackerData


class MouseTracker(EyeTracker):

    name = 'Mouse Tracker'

    def __init__(self):
        super().__init__()

        self.clock_event = None

    def start_eye_tracker(self):
        # callback ever 0.03 seconds
        Clock.schedule_interval(self.mouse_tracker_callback, 0.03)

    def stop_eye_tracker(self):
        try:
            self.clock_event.cancel()
        except AttributeError:
            pass
        self.clock_event = None

    def get_mouse_data(self):
        # assumes main monitor to be tracker monitor
        displays = screeninfo.get_monitors()
        display = list(filter(lambda x: x.is_primary, displays))[0]
        pos = pyautogui.position()
        # scale to coordinates of selected display
        gaze_selected_display_pixel = (pos[0] - display.x, pos[1] - display.y)
        # scale to normalized coordinates
        gaze_normalized = (gaze_selected_display_pixel[0] / display.width, gaze_selected_display_pixel[1] / display.height)

        return gaze_normalized
    
    def mouse_tracker_callback(self, *args):
        mouse_data = self.get_mouse_data()
        data = TrackerData()

        data.timestamp = {
            'system': int(time()*1000000)
        }
        data.left_eye = {
            'gaze_point_on_display_area': (mouse_data[0], mouse_data[1]),
            'pupil_diameter': 1
        }
        data.right_eye = {
            'gaze_point_on_display_area': (mouse_data[0], mouse_data[1]),
            'pupil_diameter': 1
        }

        self.gaze_data_callback(data)

tracker_class = MouseTracker