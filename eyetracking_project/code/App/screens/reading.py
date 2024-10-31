# kivy imports
from kivy.uix.screenmanager import Screen
from kivy.config import Config
from widgets import Overlay
from kivy.logger import Logger
from kivy.core.window import Window
# local imports
from resources import app_paths

# Read the config file
Config.read('config.ini')
PDF_PATH = app_paths.DATA_PATH / 'pdfs' / Config.get('study', 'pdf_path')

class ReadingScreen(Screen):
    '''
    Reading screen class.
    '''

    def __init__(self, **kwargs):
        super(ReadingScreen, self).__init__(**kwargs)
        self.logger = Logger
        self.recording_gaze = False
        self.current_article_id = '0'
        
        # register overlays
        self.left_eye_trackpoint = self.register_overlay()
        self.right_eye_trackpoint = self.register_overlay()
        self.avg_eye_trackpoint = self.register_overlay()
        self.display_overlay = self.register_overlay()

    def register_overlay(self):
        '''
        Register the overlay on the screen.
        '''
        overlay = Overlay()
        overlay.size = self.size
        self.add_widget(overlay)
        return overlay

    def on_enter(self):
        '''
        Callback function when the screen is entered.
        '''
        # load pdf
        pdf_path = str(PDF_PATH)
        self.logger.debug("Updating view with pdf %s", pdf_path)
        self.pdf.open(pdf_path)

    def gaze_recording(self, *args):
        '''
        Toggle the gaze recording.
        '''
        if self.ids.gaze_recording_toggle.state == 'down':
            # start recording
            self.ids.gaze_recording_toggle.text = "Stop Recording"
            self.recording_gaze = True
        else:
            # stop recording
            self.ids.gaze_recording_toggle.text = "Start Recording"
            self.recording_gaze = False
            
    def on_gaze(self, gaze, *args):
        '''
        Callback function when the screen is gazed at.

        :param gaze: The gaze data object.
        '''

        # if current screen is focused, use gaze data
        if self.manager.current == 'reading':
            left_gaze = self.transform_gaze(gaze.left_eye['gaze_point_on_display_area'])
            right_gaze = self.transform_gaze(gaze.right_eye['gaze_point_on_display_area'])
            gaze.left_eye['gaze_point_in_window'] = left_gaze
            gaze.right_eye['gaze_point_in_window'] = right_gaze
            # only record gaze data if recording is toggled
            if self.recording_gaze:
                self.pdf.on_gaze_data(gaze)
            # if gaze_view_toggle button is toggled
            if self.ids.gaze_view_toggle.state == 'down':
                self.show_gaze_points(left_gaze, right_gaze)

    def on_key_down(self, keyboard, keycode, text, modifiers):
        '''
        Callback function when a key is pressed.

        :param keyboard: The keyboard object.
        :param keycode: The keycode of the key that was pressed.
        :param text: The text of the key that was pressed.
        :param modifiers: The modifiers that were pressed.
        '''
        self.pdf.on_keypress(keycode)

    def clear_gaze_points(self, *args):
        '''
        Clear the gaze points from the screen.
        '''

        self.left_eye_trackpoint.clear_canvas()
        self.right_eye_trackpoint.clear_canvas()
        self.avg_eye_trackpoint.clear_canvas()


    def transform_gaze(self, gaze_normalized):
        '''
        Transform the gaze data to the screen coordinates.

        :param gaze_normalized: The normalized gaze data.
        '''

        d = self.manager.get_screen('screen_selector').get_selected_display()[0]

        gaze_selected_display_pixel = (d.measured_width * gaze_normalized[0], d.measured_height * gaze_normalized[1])
        gaze_display_coordinates = (d.measured_x + gaze_selected_display_pixel[0], d.measured_y + gaze_selected_display_pixel[1])
        # now we have display coordinates and need to convert them to window coordinates
        gaze_window_coordinates = (
            ((gaze_display_coordinates[0] - Window.left)),
            (Window.size[1] - (gaze_display_coordinates[1] - Window.top))
            )
        # gaze_window_normalized = (
        #         gaze_window_coordinates[0] / Window.size[0],
        #         gaze_window_coordinates[1] / Window.size[1]
        #         )
        return gaze_window_coordinates


    def show_gaze_points(self, left_gaze, right_gaze):
        '''
        Debug function for when the screen is gazed. Draws a circle at the gaze coordinates.

        :param left_gaze: The gaze coordinates of the left eye.
        :param right_gaze: The gaze coordinates of the right eye.
        '''

        size = 30
        self.left_eye_trackpoint.trackpoint(left_gaze[0], left_gaze[1], color=(1, 0, 0, 0.5), diameter=size)
        self.right_eye_trackpoint.trackpoint(right_gaze[0], right_gaze[1], color=(0, 0, 1, 0.5), diameter=size)
        self.avg_eye_trackpoint.trackpoint((left_gaze[0] + right_gaze[0]) / 2, (left_gaze[1] + right_gaze[1]) / 2, color=(0, 1, 0, 1), diameter=size/2)


    def on_leave(self, *args):
        '''
        Callback function when the screen is exited.
        '''
        pass
    