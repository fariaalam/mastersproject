from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.uix.widget import Widget

class Overlay(Widget):
    '''
    Overlay class.
    Enables drawing on top of other widgets (Like a Screen).
    '''

    def __init__(self, **kwargs):
        super(Overlay, self).__init__(**kwargs)
        with self.canvas:
            Color(0, 0, 0, 0)
            self.rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def draw_point(self, x, y, color=(0, 1, 0, 0.5), diameter=30):
        '''
        Draw a point at the given coordinates.

        :param x: The x coordinate.
        :param y: The y coordinate.
        :param color: The color of the point (default green, half transparent).
        :param diameter: The diameter of the point (default 30).
        '''
        # Draw a circle at the given coordinates
        with self.canvas.before:
            Color(*color)
            Ellipse(pos=(x - diameter / 2, y - diameter / 2), size=(diameter, diameter))

    def draw_line(self, p1, p2, color=(0, 1, 0, 0.5), width=5):
        '''
        Draw a line between two points.

        :param p1: The first point.
        :param p2: The second point.
        :param color: The color of the line (default green, half transparent).
        :param width: The width of the line (default 5).
        '''
        with self.canvas.before:
            Color(*color)
            Line(points=[*p1, *p2], width=width)

    def clear_canvas(self):
        '''
        Clear the canvas.
        '''
        self.canvas.before.clear()

    def trackpoint(self, x, y, color=(0, 1, 0, 0.5), diameter=30):
        '''
        Draw a trackpoint at the given coordinates.

        :param x: The x coordinate.
        :param y: The y coordinate.
        :param color: The color of the trackpoint (default green, half transparent).
        :param diameter: The diameter of the trackpoint (default 30).
        '''
        self.clear_canvas()
        self.draw_point(x, y, color, diameter)