# kivy imports
from kivy.logger import Logger
# other imports
import os
import importlib

class EyeTracker(object):
    '''
    Base class for eye trackers.
    '''

    name = 'Base Eye Tracker'

    def __init__(self):
        self.eyetracker = None
        # Callback function called on new gaze data
        self.gaze_data_callback = lambda x: None

    def has_tracker(self):
        '''
        Method to check if the eye tracker is connected.

        :return: True if the eye tracker is connected, False otherwise.
        '''
        return self.eyetracker is not None

    def start_eye_tracker(self):
        '''
        Start the eye tracker.
        '''
        pass

    def stop_eye_tracker(self):
        '''
        Stop the eye tracker.
        '''
        pass

tracker_class = EyeTracker

class TrackerData(object):
    '''
    Class for eye tracker data.
    '''
    def __init__(self):
        self.timestamp = {
            'system': None
        }
        self.left_eye = {
            'gaze_point_on_display_area': None,
            'pupil_diameter': None
        }
        self.right_eye = {
            'gaze_point_on_display_area': None,
            'pupil_diameter': None
        }
        self.misc = {}

    def get_data(self, datastorage, dataname):
        '''
        Access the data safely.

        :param datastorage: The attribute to access.
        :param dataname: The key of the data to access.

        :return: The data if it exists, None otherwise.
        '''
        try:
            data = datastorage[dataname]
            return data
        except KeyError:
            return None
        
    def get_common_dictionary(self):
        '''
        Get a dictionary with all the data.

        :return: A dictionary with all the data.
        '''
        common_dictionary = {}
        for i in self.timestamp:
            common_dictionary[f'timestamp {i}'] = self.get_data(self.timestamp, i)
        for i in self.left_eye:
            common_dictionary[f'left eye {i}'] = self.get_data(self.left_eye, i)
        for i in self.right_eye:
            common_dictionary[f'right eye {i}'] = self.get_data(self.right_eye, i)
        for i in self.misc:
            common_dictionary[f'misc {i}'] = self.get_data(self.misc, i)
        return common_dictionary
    
def attempt_import():
    '''
    Attempt to import all custom eye trackers.
    '''
    imported_trackers = {}

    # for all trackers in the eyeTracker directory attempt to import them
    for tracker in os.listdir(os.path.join(os.path.dirname(__file__))):
        if tracker == '__init__.py' or tracker == 'base_tracker.py':
            continue
        if tracker.endswith('.py'):
            try:
                module = importlib.import_module(f'eyeTracker.{tracker[:-3]}')
                # import class from module that is instance of EyeTracker
                try:
                    tracker_class = module.tracker_class
                    imported_trackers[tracker_class.name] = tracker_class
                except AttributeError:
                    Logger.info(f'Could not find tracker_class in {tracker}.')
            except ImportError:
                Logger.info(f'Could not import {tracker}.')

    Logger.debug(f'Imported {len(imported_trackers)} custom eye trackers.')
    for i in imported_trackers:
        Logger.debug(f" - {i}")

    return imported_trackers