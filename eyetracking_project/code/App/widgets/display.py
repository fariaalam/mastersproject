from kivy.uix.button import Button
from kivy.graphics import Color, Line

class DisplayView(Button):
    '''
    Display view class.
    '''

    def __init__(self, text='', size=(100,100), pos=(0,0), pos_adj=(0,0), **kwargs):
        super(DisplayView, self).__init__(**kwargs)
        self.text = text
        self.size_hint = (100, 100)
        self.pos_hint = {'x': 0, 'y': 0}
        self.is_selected = False
        self.measured_width, self.measured_height = size
        self.measured_x, self.measured_y = pos
        self.x_adj, self.y_adj = pos_adj
        
        with self.canvas.after:
            self.selection_color = Color(0, 0, 0, 0)
            self.selection_outline = Line(
                    rectangle=(0, 0, 0, 0),
                    width=2
                )
        # have to bind bc. real variables take time to update, when hints are change
        self.bind(size=self.update_selection_outline)
        self.bind(pos=self.update_selection_outline)
            
    def update_pos_and_size(self, scale_x, scale_y, total_x, total_y):
        '''
        Update the position and size of the display view.

        :param scale_x: The scale factor for the x-axis.
        :param scale_y: The scale factor for the y-axis.
        :param total_x: The total x-axis size.
        :param total_y: The total y-axis size.
        '''
        # offset to center
        center_x_offset = (1 - scale_x) / 2
        center_y_offset = (1 - scale_y) / 2

        # resize and reposition display
        self.size_hint=(
                (self.measured_width / total_x) * scale_x, 
                (self.measured_height / total_y) * scale_y
                )
        self.pos_hint={
                'x': ((self.x_adj) / total_x) * scale_x + center_x_offset, 
                'y': ((self.y_adj) / total_y) * scale_y + center_y_offset
                }
        
    def update_selection_outline(self, *args):
        '''
        Update the selection outline.
        '''
        # update selection outline
        self.selection_outline.rectangle = (
                self.x+7, 
                self.y+7, 
                self.width-7, 
                self.height-7
            )
        
    def update_selection(self, is_selected=False):
        '''
        Update the selection of the display view.
        
        :param is_selected: The selection state (by default False).
        '''
        self.is_selected = is_selected
        # update color of selection line
        self.selection_color.rgba = (1, 0, 0, 1 if self.is_selected else 0)