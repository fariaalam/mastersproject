# kivy imports
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from widgets import DisplayView
from kivy import Logger
# other imports
import screeninfo

class DisplayManager(object):
    '''
    Display manager class.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display_values = []

    def update_displays(self):
        '''
        update display values
        '''

        self.display_values = screeninfo.get_monitors()
    
    def get_max_min_values(self):
        '''
        get max and min values for x and y display coordinates.

        :return: max_x, max_y, min_x, min_y
        '''

        displays = screeninfo.get_monitors()
        max_x = 0
        max_y = 0
        min_x = 0
        min_y = 0
        for display in displays:
            if display.x + display.width > max_x:
                max_x = display.x + display.width
            if display.y + display.height > max_y:
                max_y = display.y + display.height
            if display.x < min_x:
                min_x = display.x
            if display.y < min_y:
                min_y = display.y
        return max_x, max_y, min_x, min_y

    def debug(self):
        '''
        print debug information
        '''

        # log current screen size
        Logger.info(f"Current screen size: {Window.size}")
        # log current screen position
        Logger.info(f"Current screen position: {Window.left}, {Window.top}")


class ScreenSelectorScreen(Screen):
    '''
    Screen selector screen class.
    '''

    def __init__(self, **kwargs):
        super(ScreenSelectorScreen, self).__init__(**kwargs)
        self.displays = []
        self.display_manager = DisplayManager()
        Window.bind(on_resize=self.on_resize)
        # register buttons
        self.ids.reset_and_reload_button.bind(on_press=self.reset_displays)

        # update monitors on creation
        if len(self.displays) < 1:
            self.reset_displays()

    def on_resize(self, *args):
        '''
        Callback function when the window is resized.
        '''
        # update monitors on resize if this screen is active
        if self.manager.current == self.name:
            self.update_monitors()

    def update_display_selection(self, *args):
        '''
        Method to update selection of displays.

        :param args: The arguments passed by the button press.
        (the first argument is the display object that was pressed)
        '''
        for display in self.displays:
            display.update_selection(False)
        args[0].update_selection(True)

    def get_selected_display(self):
        '''
        Method to get the selected display.
        
        :return: The selected displays.
        '''
        selected_displays = []
        for display in self.displays:
            if display.is_selected:
                selected_displays.append(display)
        return selected_displays
    
    def fill_display_view(self):
        '''
        Method to fill the display view with the current display values.
        '''
        # reset displays
        max_x, max_y, min_x, min_y = self.display_manager.get_max_min_values()
        self.ids.displays.clear_widgets()
        self.displays = []
        # add displays
        for e, display in enumerate(self.display_manager.display_values):
            if display.name is None:
                display.name = f'Display {e+1}'
            display_object = DisplayView(
                    text=str(f'{display.name}\n({display.width}x{display.height})'),
                    size=(display.width, display.height),
                    pos=(display.x, display.y),
                    pos_adj=(display.x - min_x, display.y - min_y)
                    )
            self.ids.displays.add_widget(display_object)
            display_object.bind(on_press=self.update_display_selection)
            if display.is_primary:
                display_object.update_selection(True)
            self.displays.append(display_object)

    def update_monitors(self):
        '''
        Method to update the monitors (sizes etc.).
        '''
        # calculate total x and y for scaling
        max_x, max_y, min_x, min_y = self.display_manager.get_max_min_values()
        total_x = max_x - min_x
        total_y = max_y - min_y

        # ensure ratio is correct
        parent_x, parent_y = self.ids.displays.size
        scale_factor = min(parent_x/total_x, parent_y/total_y)
        scale_x = (total_x * scale_factor) / parent_x
        scale_y = (total_y * scale_factor) / parent_y

        # set size of element containing displays
        sh = (1, 1)
        sh = (sh[0] * .8, sh[1] * .8)
        self.ids.displays.size_hint = sh

        # update displays
        for display_object in self.displays:
            display_object.update_pos_and_size(scale_x, scale_y, total_x, total_y)

    def reset_displays(self, *args):
        '''
        delete old displays and load new ones
        '''
        # update current display values
        self.display_manager.update_displays()
        # fill display view according to current display values
        self.fill_display_view()
        # update display sizes
        self.update_monitors()

    def on_enter(self):
        '''
        update monitors on entering the screen, to ensure correct scaling
        '''
        self.update_monitors()