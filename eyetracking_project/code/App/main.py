from kivy import Logger, LOG_LEVELS
from kivy._event import partial
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.config import Config

import os

from screens import MainMenuScreen, ReadingScreen, ScreenSelectorScreen, AnalysisScreen
from resources import app_paths
from eyeTracker.base_tracker import attempt_import


# Read the config file
Config.read('config.ini')

# Load the .kv design files from the kv directory
kv_path = str(app_paths.KV_PATH)

# Check if debug mode is enabled and load the debug.kv file
if Config.get('settings', 'debug') == 'True':
    Builder.load_file(os.path.join(kv_path, 'debug.kv'))
    DEBUG = True
    Logger.setLevel(LOG_LEVELS["debug"])
    Logger.debug("Debug mode enabled.")
else:
    DEBUG = False
    Logger.setLevel(LOG_LEVELS["info"])

# First load the widget kv file
Builder.load_file(os.path.join(kv_path, 'widgets.kv'))

# Then load all screen files
for kv_file in os.listdir(kv_path):
    # skip widgets, debug files
    if kv_file in ['widgets.kv', 'debug.kv']:
        continue
    if kv_file.endswith('.kv'):
        Builder.load_file(os.path.join(kv_path, kv_file))

# try to import eye trackers
attempt_import()
# load specified eye tracker
tracker_type = Config.get('settings', 'tracker_type')
if tracker_type == 'tobii':
    Logger.debug("Using Tobii Eye Tracker")
    from eyeTracker.tobii_tracker import TobiiTracker as EyeTracker
elif tracker_type == 'mouse':
    Logger.debug("Using Mouse Eye Tracker")
    from eyeTracker.mouse_tracker import MouseTracker as EyeTracker
else:
    raise ValueError("Invalid tracker type specified in config.ini")


class MyKivyApp(App):
    '''
    Main Kivy App class.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the eye tracker
        self.eyetracker = EyeTracker()
        self.eyetracker.gaze_data_callback = self.gaze_data_callback
        self.eyetracker.start_eye_tracker()
        # Initialize ScreenManager
        self.sm = ScreenManager()
        # create a keyboard listener
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self.root)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def build(self):
        '''
        Build the app.
        '''

        # Add all the screens to the manager
        self.sm.add_widget(MainMenuScreen(name='main_menu'))
        self.sm.add_widget(ReadingScreen(name='reading'))
        self.sm.add_widget(ScreenSelectorScreen(name='screen_selector'))
        self.sm.add_widget(AnalysisScreen(name='analysis'))

        Logger.info("Starting the App.")

        # Return the ScreenManager
        return self.sm
    
    def _keyboard_closed(self):
        '''
        Handle the keyboard being closed.
        '''
        Logger.debug('Keyboard connection lost.')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        '''
        Handle the keyboard being pressed.

        :param keyboard: The keyboard object.
        :param keycode: The keycode of the key that was pressed.
        :param text: The text of the key that was pressed.
        :param modifiers: The modifiers that were pressed.

        :return: True if the event was handled, False otherwise.
        '''
        # check if current screen has a keyboard function
        if hasattr(self.sm.current_screen, 'on_key_down'):
            self.sm.current_screen.on_key_down(keyboard, keycode, text, modifiers)
        return True

    def gaze_data_callback(self, tracker_data):
        '''
        Gaze data callback function for an eye tracker.

        :param gaze_data: The TrackerData object.
        '''
        # check if the current screen has a gaze function
        if hasattr(self.sm.current_screen, 'on_gaze'):
            Clock.schedule_once(partial(self.sm.current_screen.on_gaze, tracker_data))

    def on_stop(self):
        Logger.info("Stopping the App.")
        self.eyetracker.stop_eye_tracker()

if __name__ == '__main__':
    MyKivyApp().run()
