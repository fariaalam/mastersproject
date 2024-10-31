import fitz
from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
import os
from datetime import datetime

# TODO: store pdf pages in textures for kivy instead of as a file in the directory
class PdfDocument(object):
    '''
    A class to represent a pdf document.
    '''
    def __init__(self, path):
        self.path = path
        try:
            self.pdf = fitz.open(path) if path.endswith('.pdf') else None
        except FileNotFoundError:
            self.pdf = None
        self.pages = self.pdf.page_count if self.pdf else 0
        self.cache_dir = './widgets/pdf_view_cache'

    def get_page(self, page_number):
        '''
        :param page_number: The page number to load.
        :return: The page object if it exists, None otherwise.
        '''
        return self.pdf.load_page(page_number) if self.pdf else None
    
    def get_page_size(self, page_number):
        '''
        :param page_number: The page number to load.
        :return: The size of the page.
        '''
        page = self.get_page(page_number)
        return page.rect.width, page.rect.height
    
    def get_page_normalized_size(self, page_number):
        '''
        :param page_number: The page number to load.
        :return: The normalized size of the page (the smaller side is 1, the other one scaled relatively).
        '''
        width, height = self.get_page_size(page_number)
        min_size = min(width, height)
        return width / min_size, height / min_size

    def close(self):
        '''
        Closing the pdf file.
        '''
        self.pdf.close() if self.pdf else None
        # TODO remove cache

    def image_of_page(self, page_number, matrix=fitz.Matrix(1, 1)):
        '''
        :param page_number: The page number to load.
        :param matrix: The matrix to scale the image with. (by default Identity)
        :return: The path to the image of the page.
        '''
        pixmap = self.get_page(page_number).get_pixmap(matrix=matrix) #dpi=600
        # save image
        save_path = f'{self.cache_dir}/{page_number}.png'
        pixmap.save(save_path, 'png')
        return save_path
    
# Kivy objects
    
class PdfPage(AnchorLayout):
    '''
    A widget to display a single page of a pdf document.
    '''

    source = StringProperty(None)

    def __init__(self, page: int, pdf_document: PdfDocument, log_dir: str, **kwargs):
        '''
        :param page: The page number to display.
        :param pdf_document: The PdfDocument object to display.
        :param log_dir: The directory to save the log file to. 
        '''
        super(PdfPage, self).__init__(**kwargs)
        self.page = page
        self.pdf = pdf_document
        self.log_dir = log_dir
        self.log_file = f'{self.log_dir}/{self.page+1}.log'
        self.scaling_factor = 1
        self.bind(size=self.update_size)

    def update_size(self, *args):
        # if no source to load image yet, return
        if not self.source:
            return
        # calculate scaling factor
        x_fac = (self.size[0] / self.image.texture_size[0])
        y_fac = (self.size[1] / self.image.texture_size[1])
        min_fac = min(x_fac, y_fac)
        self.scaling_factor *= min_fac if min_fac > 0 else 1
        # update image with new scaling factor (reload image)
        self.load_page()

    def load_page(self):
        self.source = self.pdf.image_of_page(
            self.page, 
            matrix=fitz.Matrix(self.scaling_factor, self.scaling_factor)
            )
        self.image.reload()

    def normalize_gaze(self, gaze_point):
        return (
                (gaze_point[0] - self.image.pos[0]) / self.image.size[0],
                (gaze_point[1] - self.image.pos[1]) / self.image.size[1]
                )

    def on_gaze(self, gaze):
        left_eye = gaze.left_eye['gaze_point_in_window']
        right_eye = gaze.right_eye['gaze_point_in_window']
        # detect if gaze is in page
        if self.image.collide_point(left_eye[0], left_eye[1]) or self.image.collide_point(right_eye[0], right_eye[1]):
            #print(f'gaze on image: {left_eye}, {right_eye}, {self.pos}, {self.image.size} {self.image.pos}')
            # log the data to a file
            self.log_data(gaze)

    def log_data(self, gaze):
        # determine normalized pdf gaze points (for range 0-1)
        left_normalized = self.normalize_gaze(gaze.left_eye['gaze_point_in_window'])
        right_normalized = self.normalize_gaze(gaze.right_eye['gaze_point_in_window'])
        #print(f'normalized gaze: {left_normalized}, {right_normalized}')
        datapoints = [
            f'{gaze.timestamp["system"]}',
            f'{left_normalized[0]}',
            f'{left_normalized[1]}',
            f'{right_normalized[0]}',
            f'{right_normalized[1]}'
        ]
        self.log_file.write(','.join(datapoints) + '\n')

    def open(self):
        # open log file
        self.log_file = open(f'{self.log_file}', 'a')
        # load pdf page
        self.load_page()
        # scale to size as soon as pdf is loaded
        self.update_size()

    def close(self):
        self.log_file.close()


class PdfPageOverlay(Image):
    def __init__(self, pdf_page: PdfPage, **kwargs):
        super(PdfPageOverlay, self).__init__(**kwargs)
        self.bind(pdf_page.size, self.update_size)
        self.opacity = 0.5

    def update_size(self, *args):
        self.size = self.pdf_page.size

    def load_picture(self, picture_path):
        self.source = picture_path
        self.reload()


class PdfPageView(BoxLayout):
    def __init__(self, **kwargs):
        super(PdfPageView, self).__init__(**kwargs)
        self.pdf = None
        self.pdf_page = None
        self.log_dir = None
        self.current_page = 0

    def on_gaze_data(self, gaze_data):
        if self.pdf_page:
            self.pdf_page.on_gaze(gaze_data)

    def on_keypress(self, keycode):
        '''
        Handle keypress events.
        
        :param keycode: The keycode of the pressed key.
        '''
        # if right arrow key, show next page
        if keycode[1] == 'right':
            self.show_next_page()
        # if left arrow key, show previous page
        elif keycode[1] == 'left':
            self.show_prev_page()

    def open(self, path):
        '''
        Try to open the pdf file at the given path.

        :param path: The path to the pdf file.
        :return: True if the file was opened successfully, False otherwise.
        '''
        # close any open pdf
        self.close()
        # try to open the pdf
        self.pdf = PdfDocument(path)
        # try to create a log folder
        self.log_dir = f'{os.path.dirname(self.pdf.path)}/logfiles_{os.path.basename(self.pdf.path)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.pdf.pages > 0:
            self.show_page(0)
            return True
        else:
            return False

    def show_page(self, page_number):
        # check if page number is valid
        if page_number < 0 or page_number >= self.pdf.pages:
            return
        # update current page
        self.current_page = page_number
        # update pdf page
        self.update_pdf_page()

    def update_pdf_page(self):
        # remove old pdf page
        if self.pdf_page:
            self.pdf_page.close()
            self.pdf_page_container.remove_widget(self.pdf_page)
        # create new pdf page
        self.pdf_page = PdfPage(self.current_page, self.pdf, self.log_dir)
        self.pdf_page.open()
        self.pdf_page_container.add_widget(self.pdf_page)
        #self.add_widget(self.pdf_page)
        #page.open(image_path)

    def show_next_page(self):
        if self.pdf:
            self.show_page(self.current_page + 1)

    def show_prev_page(self):
        if self.pdf:
            self.show_page(self.current_page - 1)

    def close(self):
        # if pdf is open, close it
        if self.pdf:
            self.pdf.close()
        self.pdf = None


# TODO: Implement the PdfScrollView class
# hint: kivy provides a ScrollView widget
class PdfScrollView(BoxLayout):
    def __init__(self, **kwargs):
        super(PdfScrollView, self).__init__(**kwargs)
        self.pdf = None
        self.log_dir = None


    def on_gaze_data(self, gaze_data):
        pass

    def on_keypress(self, key):
        pass

    def open(self, path):
        '''
        Try to open the pdf file at the given path.

        :param path: The path to the pdf file.
        '''

        # close any open pdf
        self.close()
        # try to open the pdf
        self.pdf = PdfDocument(path)
        # try to create a log folder
        self.log_dir = f'{os.path.dirname(self.pdf.path)}/logfiles_{os.path.basename(self.pdf.path)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.pdf.pages > 0:
            self.load_page(0)
            self.load_page(0)
            self.load_page(0)
            self.load_page(0)
            self.load_page(0)
            return True
        else:
            return False
        
    def load_page(self, page_number):
        pdf_page = PdfPage(page_number, self.pdf, self.log_dir)
        pdf_page.open()
        self.pdf_page_container.add_widget(pdf_page)
        
    def close(self):
        # if pdf is open, close it
        if self.pdf:
            self.pdf.close()
        self.pdf = None