from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

class FilesystemBrowser(FloatLayout):

    def __init__(self, **kwargs):
        super(FilesystemBrowser, self).__init__(**kwargs)
        self._popup = Popup(
            title="Filesystem Browser",
            content=self,
            size_hint=(0.9, 0.9)
            )

    def close(self, *args):
        print(args)
        self._popup.dismiss()

    def show(self, *args):
        print(args)
        self._popup.open()

    def select(self, *args):
        print(args)