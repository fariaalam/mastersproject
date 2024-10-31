# kivy imports
from kivy.uix.screenmanager import Screen
import fitz
from kivy.config import Config
from widgets.filesystem import FilesystemBrowser
from resources import app_paths
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import numpy as np

from utils.heatmap_opacity import (
    get_heatmap
)

from utils.metrics import (
    calculate_fixation_time_AOIs, 
    caluclate_fixation_count_AOIs,
    caluclate_average_fixation_duration_AOIs,
    calculate_roaft_aois,
    calculate_progreg_rate_aoi,
    readig_pattern_analysis
)

from utils.aoi import (
    detect_area,
    get_aoi_coverages
)

from utils.fixation_saccades import (
    get_fixations_from_gaze,
    calculate_saccades,
    calculate_forward_saccades,
    calculate_backward_saccades,
    aoi_saccades_counts,
    calculate_saccades_length_AOIs
)

# Read the config file
Config.read('config.ini')
PDF_PATH = app_paths.DATA_PATH / 'pdfs' / Config.get('study', 'pdf_path')

class AnalysisScreen(Screen):
    '''
    Analysis screen class.
    '''

    def __init__(self, **kwargs):
        super(AnalysisScreen, self).__init__(**kwargs)
        self.display_size = None
        self.filebrowser = FilesystemBrowser()
        self.image_cahce_dir = './widgets/pdf_view_cache/'
        self.current_page = 0
        self.current_image_path = None
        self.current_clicked = None
        self.last_created_image = None
        self.fixation_clicked = None

    def on_enter(self):
        self.log_folders = self.get_log_folders()
        #self.log_folders = log_folders
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()
        image = Image.open(self.current_image_path)
        width, height = image.size
        self.display_size = (width, height)
        doc = fitz.open(PDF_PATH)
        self.page_count = doc.page_count
        image_path = str(os.getcwd())+"/"+self.current_image_path
        self.aois = detect_area(image_path, self.display_size, level = 3, aoi_type = "paragraph")
        self.para_informations, self.user_pattern_information, self.analysis_report = self.get_all_para_information(self.aois)
      


    def get_log_folders(self):
        # get the latest four folders from the "log path" directory contaning the pdf name
        pdf_name = str(Config.get('study', 'pdf_path'))
        log_path = str(app_paths.DATA_PATH)+'/pdfs/'
        entries = os.listdir(log_path)
        directories = [d for d in entries if os.path.isdir(os.path.join(log_path, d))]
        sorted_directories = sorted(directories, key=lambda x: os.path.getmtime(os.path.join(log_path, x)), reverse=True)
        pdf_log_folders = []
        users = 0
        for idx, folder_name in enumerate(sorted_directories):
            
            if pdf_name in folder_name:
                folder_name = log_path + str(folder_name)
                pdf_log_folders.append(folder_name)
                users = users + 1
                if users == 4:
                    break
        
        return pdf_log_folders



    def open_filebrowser(self):
        '''
        Open the file browser.
        '''
        self.filebrowser.show()
    
    def change_toggle_user1(self):
        # function called when reader 1 toggle button is clicked
        self.ids.toggle_user_2.state = "normal"
        self.ids.toggle_user_3.state = "normal"
        self.ids.toggle_user_4.state = "normal"
        self.ids.combined.state = "normal"
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        self.fixation_clicked = None
        self.log_file = f"{self.log_folders[0]}/{self.current_page+1}.log"
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()
        
    
    def change_toggle_user2(self):
        # function called when reader 2 toggle button is clicked
        self.ids.toggle_user_1.state = "normal"
        self.ids.toggle_user_3.state = "normal"
        self.ids.toggle_user_4.state = "normal"
        self.ids.combined.state = "normal"
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        self.fixation_clicked = None
        self.log_file = f"{self.log_folders[1]}/{self.current_page+1}.log"
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()
        
    
    def change_toggle_user3(self):
        # function called when reader 3 toggle button is clicked
        self.ids.toggle_user_1.state = "normal"
        self.ids.toggle_user_2.state = "normal"
        self.ids.toggle_user_4.state = "normal"
        self.ids.combined.state = "normal"
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        self.fixation_clicked = None
        self.log_file = f"{self.log_folders[2]}/{self.current_page+1}.log"
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()
        

    def change_toggle_user4(self):
        # function called when reader 4 toggle button is clicked
        self.ids.toggle_user_1.state = "normal"
        self.ids.toggle_user_2.state = "normal"
        self.ids.toggle_user_3.state = "normal"
        self.ids.combined.state = "normal"
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        self.fixation_clicked = None
        self.log_file = f"{self.log_folders[3]}/{self.current_page+1}.log"
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()
        

    def change_toggle_combined(self):
        # function called when combined toggle button is clicked
        self.ids.toggle_user_1.state = "normal"
        self.ids.toggle_user_2.state = "normal"
        self.ids.toggle_user_3.state = "normal"
        self.ids.toggle_user_4.state = "normal"
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        self.fixation_clicked = None
        self.ids.image1.source = self.current_image_path
        self.ids.image1.reload()

    def show_prev_page(self):
        # function called when previous button is clicked
        prev_page = self.current_page
        if prev_page  > 0:
            prev_page = prev_page - 1
        if prev_page != self.current_page:
            self.current_page = prev_page
            self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
            self.ids.image1.source = self.current_image_path
            self.ids.image1.reload()
            self.ids.toggle_user_1.state = "normal"
            self.ids.toggle_user_2.state = "normal"
            self.ids.toggle_user_3.state = "normal"
            self.ids.toggle_user_4.state = "normal"
            self.ids.combied.state = "normal"
            self.fixation_clicked = None
            self.log_file = None
            self.current_clicked = None
            self.ids.report.text = "Analysis Report"
            self.ids.vbox.clear_widgets()
            image_path = str(os.getcwd())+"/"+self.current_image_path
            self.aois = detect_area(image_path, self.display_size, level = 3, aoi_type = "paragraph")
            self.para_informations, self.user_pattern_information, self.analysis_report = self.get_all_para_information(self.aois)


    def show_next_page(self):
        # function called when next button is clicked
        next_page = self.current_page
        if (next_page+1)  < self.page_count:
            next_page = next_page + 1
        
        if next_page != self.current_page:
            self.current_page = next_page
            self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
            self.ids.image1.source = self.current_image_path
            self.ids.image1.reload()
            self.ids.toggle_user_1.state = "normal"
            self.ids.toggle_user_2.state = "normal"
            self.ids.toggle_user_3.state = "normal"
            self.ids.toggle_user_4.state = "normal"
            self.ids.combied.state = "normal"
            self.fixation_clicked = None
            self.current_clicked = None
            self.log_file = None
            self.ids.report.text = "Analysis Report"
            self.ids.vbox.clear_widgets()
            image_path = str(os.getcwd())+"/"+self.current_image_path
            self.aois = detect_area(image_path, self.display_size, level = 3, aoi_type = "paragraph")
            self.para_informations, self.user_pattern_information, self.analysis_report = self.get_all_para_information(self.aois)

    def refresh(self):
        # function called when refresh button is clicked
        self.current_image_path = f'./widgets/pdf_view_cache/{self.current_page}.png'
        self.ids.image1.source = self.current_image_path
        self.ids.toggle_user_1.state = "normal"
        self.ids.toggle_user_2.state = "normal"
        self.ids.toggle_user_3.state = "normal"
        self.ids.toggle_user_4.state = "normal"
        self.ids.combined.state = "normal"
        self.fixation_clicked = None
        self.current_clicked = None
        self.ids.report.text = "Analysis Report"
        self.ids.vbox.clear_widgets()
        
    def plot_para_aoi(self, intensities, save_imgae_path, message = None):   
        # function to plot rectangles on paragraphs with various intensities 
        # for all users combined and for indivudual users as well, whether the 
        # plot will be for all users or single user is controlled by the message argument 
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        legend_dict ={
            "0.00" : "0/4 readers",
            "0.25" : "1/4 readers",
            "0.50" : "2/4 readers",
            "0.75" : "3/4 readers",
            "1.00" : "4/4 readers",
        }
        image = Image.open(self.current_image_path)
        image = image.resize(self.display_size, Image.LANCZOS)
        fig, ax = plt.subplots()
        ax.imshow(image)
        for idx, pair in enumerate(self.aois.items()):
            key, area = pair
            x, y, width, height = area
            alpha = intensities[idx]
            if message != None:
                alpha = intensities[idx]
            rect = patches.Rectangle((x,y), width, height, linewidth=0.2, edgecolor='g', facecolor=(0, 0, 1, alpha))
            ax.add_patch(rect)

        if message == None:
            legend_patches = [
                patches.Patch(color=(0, 0, 1, a), label=legend_dict[f'{a:.2f}']) for a in sorted(set(intensities)) 
            ]
            ax.legend(handles=legend_patches, loc='upper right',fontsize = "xx-small")
        else:
            legend_patches = patches.Patch(color=(0, 0, 1, 0.5), label=message) 
            ax.legend(handles=[legend_patches], loc='upper right',fontsize = "xx-small")

        if os.path.exists(save_imgae_path):
            os.remove(save_imgae_path)
        if message == None:
            self.last_created_image = save_imgae_path 
        plt.axis('off')
        plt.savefig(save_imgae_path, dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close()
        self.ids.image1.source = save_imgae_path
        self.ids.image1.reload() 

    def check_down(self):
        # function to check which reader toogle button is down 
        user_id = None
        if self.ids.toggle_user_1.state == "down":
            user_id =  0
        elif self.ids.toggle_user_2.state == "down":
            user_id =  1
        elif self.ids.toggle_user_3.state == "down":
            user_id =  2
        elif self.ids.toggle_user_4.state == "down":
            user_id =  3
        return user_id

    def user_intensities(self, user_id, pattern):
        # for a single user when read thoroughly or skimmed or skipped button is 
        # clicked the paragraphs which were read thoroughly or skimmed or skipped
        # by that user is assigned intensity 0.5 otherwise 0.0
        intensities = []
        for i in range(len(self.analysis_report[0])):
            if self.analysis_report[user_id][i]['Reading Pattern'] == pattern:
                intensities.append(0.5)
            else:
                intensities.append(0)
        return intensities
    
    def combined_text(self, pattern):
        # when read thoroughly or skimmed or skipped button is clicked and also when
        # the combined toggle button is down then we show the which paragraphs were
        # read thoroughly or skimmed or skipped in the analysis report
        sentence = "" 
        for aoi_id in range(len(self.aois)):
            found = []
            for user_id in range(len(self.log_folders)):
                if self.analysis_report[user_id][aoi_id]['Reading Pattern'] == pattern:
                    found.append(user_id)
            if len(found) > 0:
                users = ", ".join([str(i+1) for i in found])
                if pattern == "Read Thoroughly":
                    sentence = sentence + f"Paragraph {aoi_id+1} :  Reader {users} Read Thoroughly\n"
                else:
                    sentence = sentence + f"Paragraph {aoi_id+1} :  Reader {users} {pattern}\n"
        self.ids.report.text = sentence


    def show_information_for_user(self, user_id):  
        # when read thoroughly or skimmed or skipped button is clicked and also when
        # one of the reader toggle button is down then we show which paragraphs were
        # read thoroughly or skimmed or skipped by the user is there is any and also
        # show some metrics for each of the paragraphs in the given page such AFD, ROAFT
        # Coverages, number for forward and backward saccades
        sentence = ""
        total_fixation_duration = sum(self.para_informations[user_id]["fixation times per aoi"])
        total_fixation_count =  sum(self.para_informations[user_id]["fixation counts per aoi"])
        full_page_afd = total_fixation_duration/total_fixation_count
        sentence = sentence + f"Reader {user_id +1} full page AFD : {round(full_page_afd, 3)}\n\n"
        for aoi_id in range(len(self.aois)):
            user_afd = self.para_informations[user_id]["average fixation duration aoi"][aoi_id]
            user_roaft = self.para_informations[user_id]["roafts"][aoi_id]
            user_coverage = self.para_informations[user_id]["coverages"][aoi_id]
            forward_saccades_count = self.para_informations[user_id]["forward saccades per aoi"][aoi_id]
            backward_saccades_count = self.para_informations[user_id]["backward saccades per aoi"][aoi_id]

            sentence = sentence + f"Paragraph {aoi_id+1} : \n AFD : {round(user_afd,3)} ,  coverage: {user_coverage} \n"
            sentence = sentence + f"ROAFT : {user_roaft},  forward saccades : {forward_saccades_count} \n"
            sentence = sentence + f"backward saccades : {backward_saccades_count} \n\n"
        self.ids.report.text = sentence
                

    def generate_relevent(self):
        # this function is called when read thoroughly button is clicked
        self.ids.report.text = "Analysis Report"
        self.ids.vbox.clear_widgets()
        if self.ids.combined.state == "down":
            self.fixation_clicked = None
            self.current_clicked = "Read Thoroughly"
            para_user_count = [] 
            user_count = len(self.log_folders)
            for para_id, values in self.user_pattern_information.items():
                para_user_count.append(values[1])
            intensities = [n/user_count for n in para_user_count]
            save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_relevent_aoi.png'
            self.plot_para_aoi(intensities, save_image_path)
            self.combined_text(pattern="Read Thoroughly")

        else:
            user_id = self.check_down()
            if user_id != None:
                self.fixation_clicked = None
                save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_relevent_aoi.png'
                intensities = self.user_intensities(user_id=user_id, pattern="Read Thoroughly")
                self.plot_para_aoi(intensities, save_image_path, message="Read Thoroughly")
                self.show_information_for_user(user_id)

    

    def generate_skimmed(self):
        # this function is called when skimmed button is clicked
        self.ids.report.text = "Analysis Report"
        self.ids.vbox.clear_widgets()
        if self.ids.combined.state == "down":
            self.fixation_clicked = None
            self.current_clicked = "Skimmed"
            para_user_count = [] 
            user_count = len(self.log_folders)
            for para_id, values in self.user_pattern_information.items():
                para_user_count.append(values[0])
            intensities = [n/user_count for n in para_user_count]
            save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_skimmed_aoi.png'
            self.plot_para_aoi(intensities, save_image_path)
            self.combined_text(pattern="Skimmed")
        else:
            user_id = self.check_down()
            if user_id != None:
                self.fixation_clicked = None
                save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_skimmed_aoi.png'
                intensities = self.user_intensities(user_id=user_id, pattern="Skimmed")
                self.plot_para_aoi(intensities, save_image_path, message="Skimmed")
                self.show_information_for_user(user_id)
    
    def generate_skipped(self):
        # this function is called when skipped button is clicked
        self.ids.report.text = "Analysis Report"
        self.ids.vbox.clear_widgets()
        if self.ids.combined.state == "down":
            self.current_clicked = "Skipped"
            para_user_count = [] 
            user_count = len(self.log_folders)
            for para_id, values in self.user_pattern_information.items():
                para_user_count.append(values[2])
            intensities = [n/user_count for n in para_user_count]
            save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_skipped_aoi.png'
            self.plot_para_aoi(intensities, save_image_path)
            self.combined_text(pattern="Skipped")
        else:
            user_id = self.check_down()
            if user_id != None:
                self.fixation_clicked = None
                save_image_path = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_skipped_aoi.png'
                intensities = self.user_intensities(user_id=user_id, pattern="Skipped")
                self.plot_para_aoi(intensities, save_image_path, message="Skipped")
                self.show_information_for_user(user_id)


    def generate_opacity(self):
        # generate opacity of the shwo pdf image based on fixations
        # first heat map is calculated then opcaity map is created based
        # on the heatmap
        if self.check_down() != None:
            fixations = get_fixations_from_gaze(self.log_file, self.display_size)

            # get heatmap matrix of size (display_size [0], display_size[1])
            heatmap_matrix = get_heatmap(self.display_size, fixations)

            img = Image.open(self.current_image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = img.resize(self.display_size, Image.LANCZOS)

            numpy_array = np.array(img)
            for i in range(numpy_array.shape[0]):
                for j in range(numpy_array.shape[1]):
                    for k in range(numpy_array.shape[2]):
                        if np.isnan(heatmap_matrix[i][j]):
                            numpy_array[i][j][k] = 0

            img = Image.fromarray(numpy_array)
            plt.imshow(img)
            plt.axis('off')
            if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_opacity.png'):
                os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_opacity.png')
            plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_opacity.png', 
                        dpi=1000, bbox_inches='tight', pad_inches=0)
            plt.close()

            self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_opacity.png'
            self.ids.image1.reload()


    def generate_heatmap(self):
        # fuction called when heatmap button is clicked
        if self.check_down() != None:
            fixations = get_fixations_from_gaze(self.log_file, self.display_size)


            # get heatmap matrix of size (display_size [0], display_size[1])
            heatmap_matrix = get_heatmap(self.display_size, fixations)

            img = Image.open(self.current_image_path)
            img = img.resize(self.display_size, Image.LANCZOS)

            plt.imshow(img)
            ax = plt.gca()

            ax.imshow(heatmap_matrix, cmap='jet', alpha=0.5)
            plt.axis('off')
            if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap.png'):
                os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap.png')

            plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap.png', 
                        dpi=1000, bbox_inches='tight', pad_inches=0)
            plt.close()

            self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap.png'
            self.ids.image1.reload()

    def plot_fixation_user(self):
        # when any of the reader toggle buttons is down this function is called to 
        # generate fixations plot for that particular reader
        fixations = get_fixations_from_gaze(self.log_file, self.display_size)
        img = Image.open(self.current_image_path)
        img = img.resize(self.display_size, Image.LANCZOS)

        plt.imshow(img)
        ax = plt.gca()

        for idx, pair in enumerate(self.aois.items()):
            key, area = pair
            x, y, width, height = area
            rect = patches.Rectangle((x,y), width, height, linewidth=0.4, edgecolor='g', facecolor=(0, 0, 1, 0))
            ax.add_patch(rect)

        # Plot each fixation
        for idx, data in enumerate(fixations):
            _, _, duration, x, y = data
            circle = plt.Circle((x, y), duration/50, color='green', fill = True)
            ax.add_patch(circle)
            
        plt.axis('off')  # To turn off the axis
        if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png'):
            os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png')

        plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png', 
                    dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close()

        self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png'
        self.ids.image1.reload()

    def plot_fixation_combined(self):
        # when combined toggle buttons is down this function is called to 
        # generate fixations plot for all the readers
        img = Image.open(self.current_image_path)
        img = img.resize(self.display_size, Image.LANCZOS)

        plt.imshow(img)
        ax = plt.gca()

        for idx, pair in enumerate(self.aois.items()):
            key, area = pair
            x, y, width, height = area
            rect = patches.Rectangle((x,y), width, height, linewidth=0.4, edgecolor='g', facecolor=(0, 0, 1, 0))
            ax.add_patch(rect)
        for index, log_folder in enumerate(self.log_folders):

            log_file = log_folder + "/" + f"{self.current_page+1}.log"

            fixations = get_fixations_from_gaze(log_file, self.display_size)
            

            # Plot all user fixation in combined mode
            for idx, data in enumerate(fixations):
                _, _, duration, x, y = data
                if index == 0:
                    circle = plt.Circle((x, y), 3, color='green', fill = True)
                if index == 1:
                    circle = plt.Circle((x, y), 3, color='blue', fill = True)
                if index == 2:
                    circle = plt.Circle((x, y), 3, color='red', fill = True)
                if index == 3:
                    circle = plt.Circle((x, y), 3, color='purple', fill = True)
                ax.add_patch(circle)

        green_circle = plt.Line2D([0], [0], marker='o', color='w', label='Reader 1',
                            markerfacecolor='green', markersize=5)
        blue_circle = plt.Line2D([0], [0], marker='o', color='w', label='Reader 2',
                            markerfacecolor='blue', markersize=5)
        red_circle = plt.Line2D([0], [0], marker='o', color='w', label='Reader 3',
                            markerfacecolor='red', markersize=5)
        purple_circle = plt.Line2D([0], [0], marker='o', color='w', label='Reader 4',
                            markerfacecolor='purple', markersize=5)
        plt.legend(handles=[green_circle, blue_circle, red_circle, purple_circle], loc='upper right', fontsize = 'xx-small')
                
        plt.axis('off')  # To turn off the axis
        if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png'):
            os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png')

        plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png', 
                    dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close()

        self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_fixation.png'
        self.ids.image1.reload()


    def generate_fixation(self):
        # function called when fixation button is clicked
        self.ids.report.text = "Analysis Report"
        self.current_clicked = None
        if self.check_down() != None:
            self.plot_fixation_user()
        if self.ids.combined.state == "down":
            self.fixation_clicked = "Clicked"
            self.plot_fixation_combined()
            

    def generate_scanpath(self):
        # function called when scanpath button is clicked
        if self.check_down() != None:
            fixations = get_fixations_from_gaze(self.log_file, self.display_size)
            img = Image.open(self.current_image_path)
            img = img.resize(self.display_size, Image.LANCZOS)

            # collect saccades related info for adding arrows to saccades in scanpath
            saccades = calculate_saccades(fixations)
            backward_saccades = calculate_backward_saccades(saccades)
            forward_saccades = calculate_forward_saccades(saccades)
            forward_saccades_xy = [(element[6],element[7],element[8],element[9]) for element in forward_saccades]
            backward_saccades_xy = [(element[6],element[7],element[8],element[9]) for element in backward_saccades]

            plt.imshow(img)
            ax = plt.gca()

            # Plot each fixation
            for idx, data in enumerate(fixations):
                _, _, duration, x, y = data
                circle = plt.Circle((x, y), 4, color='green', fill = True)
                ax.add_patch(circle)

            for i in range(len(fixations)-1):
                _, _, _, x1, y1 = fixations[i]
                _, _, _, x2, y2 = fixations[i+1]

                forward_found_match = 0
                for sample in forward_saccades_xy:
                    a1, b1, a2, b2 = sample
                    if x1==a1 and b1==y1 and a2==x2 and b2==y2:
                        forward_found_match = 1

                backward_found_match = 0
                for sample in backward_saccades_xy:
                    a1, b1, a2, b2 = sample
                    if x1==a1 and b1==y1 and a2==x2 and b2==y2:
                        backward_found_match  = 1

                if backward_found_match == 1:
                    arrow = patches.FancyArrowPatch([x1, y1], [x2, y2], color='red', linestyle = "--", 
                                                    linewidth=0.8, arrowstyle='->', mutation_scale=20)
                    ax.add_patch(arrow)
                elif forward_found_match == 1:
                    arrow = patches.FancyArrowPatch([x1, y1], [x2, y2], color='blue', linestyle = "--", 
                                                    linewidth=0.8, arrowstyle='->', mutation_scale=20)
                    ax.add_patch(arrow)
                else:
                    ax.plot([x1, x2], [y1, y2], color='green', linestyle = "--", linewidth=0.5)
            
            green_line = plt.Line2D([], [], color='green', lw=1, linestyle='--', label='Scan Path')
            blue_arrow = plt.Line2D([], [], color='blue', lw=1, linestyle='--', label='Forward Saccade')
            red_arrow = plt.Line2D([], [], color='red', lw=1, linestyle='--', label='Backward Saccade')
            plt.legend(handles=[green_line, blue_arrow, red_arrow], loc='upper right', fontsize = 'xx-small')

            plt.axis('off')  
            if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_scanpath.png'):
                os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_scanpath.png')

            plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_scanpath.png', 
                        dpi=1000, bbox_inches='tight', pad_inches=0)
            plt.close()

            self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_scanpath.png'
            self.ids.image1.reload()


    def generate_saccades(self):
        # function called when saccades button is clicked
        if self.check_down() != None:
            fixations = get_fixations_from_gaze(self.log_file, self.display_size)
            saccades = calculate_saccades(fixations)

            img = Image.open(self.current_image_path)
            img = img.resize(self.display_size, Image.LANCZOS)

            saccades = calculate_saccades(fixations)
            backward_saccades = calculate_backward_saccades(saccades)
            forward_saccades = calculate_forward_saccades(saccades)
            forward_saccades_xy = [(element[6],element[7],element[8],element[9]) for element in forward_saccades]
            backward_saccades_xy = [(element[6],element[7],element[8],element[9]) for element in backward_saccades]

            plt.imshow(img)
            ax = plt.gca()

            for i in range(len(saccades)):
                _, _, _, _, _, _, x1, y1, x2, y2 = saccades[i]
                forward_found_match = 0
                for sample in forward_saccades_xy:
                    a1, b1, a2, b2 = sample
                    if x1==a1 and b1==y1 and a2==x2 and b2==y2:
                        forward_found_match = 1

                backward_found_match = 0
                for sample in backward_saccades_xy:
                    a1, b1, a2, b2 = sample
                    if x1==a1 and b1==y1 and a2==x2 and b2==y2:
                        backward_found_match  = 1

                if backward_found_match == 1:
                    arrow = patches.FancyArrowPatch([x1, y1], [x2, y2], color='red', linestyle = "--", 
                                                    linewidth=0.8, arrowstyle='->', mutation_scale=20)
                    ax.add_patch(arrow)
                elif forward_found_match == 1:
                    arrow = patches.FancyArrowPatch([x1, y1], [x2, y2], color='blue', linestyle = "--", 
                                                    linewidth=0.8, arrowstyle='->', mutation_scale=20)
                    ax.add_patch(arrow)
                
            # Plot each fixation
            for idx, data in enumerate(saccades):
                _, _, _, _, _, _, x1, y1, x2, y2 = saccades[idx]
                circle = plt.Circle((x1, y1), 4, color='blue', fill = True)  
                ax.add_patch(circle)
                circle = plt.Circle((x2, y2), 4, color='red', fill = True)  
                ax.add_patch(circle)

            blue_arrow = plt.Line2D([], [], color='blue', lw=1, linestyle='--', label='Forward Saccade')
            red_arrow = plt.Line2D([], [], color='red', lw=1, linestyle='--', label='Backward Saccade')
            blue_circle = plt.Line2D([0], [0], marker='o', color='w', label='Start of Saccade',
                            markerfacecolor='blue', markersize=5)
            red_circle = plt.Line2D([0], [0], marker='o', color='w', label='End of Saccade',
                                    markerfacecolor='red', markersize=5)
            plt.legend(handles=[blue_circle, red_circle, blue_arrow, red_arrow], loc='upper right', fontsize = 'xx-small')
                
            plt.axis('off')  
            if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_saccades.png'):
                os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_saccades.png')

            plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_saccades.png'
                        ,dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_saccades.png'
            self.ids.image1.reload()



    def show_combined_info(self, clicked_aoi):
        # this function is called when combined toogle button is down one the read
        # thoroughly, skimmed or skipped button is clicked and then
        # one the paragraphs is clicked, to show all information related to
        # that clicked parapgraph 
        if self.current_clicked != None:

            # reading pattern for that paragraph 
            sentence = ""
            found = []
            for user_id in range(len(self.log_folders)):
                if self.analysis_report[user_id][clicked_aoi]['Reading Pattern'] == self.current_clicked:
                    found.append(user_id)
            if len(found) > 0:
                users = ", ".join([str(i+1) for i in found])
                if self.current_clicked == "Read Thoroughly":
                    sentence = sentence + f"Paragraph  {clicked_aoi+1} :  Reader {users} Read Thoroughly\n"
                else:
                    sentence = sentence + f"Paragraph  {clicked_aoi+1} :  Reader {users} {self.current_clicked}\n"
            sentence = sentence + "\n"

            sentence = sentence + f"Combined information for paragraph {clicked_aoi+1} \n\n"
            
            # dwell time
            dwell_time = 0
            for i in range(len(self.log_folders)):
                dwell_time =  dwell_time + self.para_informations[i]['fixation times per aoi'][clicked_aoi]
            sentence = sentence + f"total dwell time : {round(dwell_time,3)} ms\n\n"


            backward_saccades = 0
            for i in range(len(self.log_folders)):
                backward_saccades =  backward_saccades + self.para_informations[i]['backward saccades per aoi'][clicked_aoi]
            #sentence = sentence + f"total backward saccades : {backward_saccades} \n\n"

            # average fixation duration of that whole page for all users
            sentence = sentence + "Readers whole page Average Fixation Duration : \n"
            for i in range(len(self.log_folders)):
                total_fixation_time = sum(self.para_informations[i]["fixation times per aoi"])
                total_fixation_count = sum(self.para_informations[i]["fixation counts per aoi"])
                user_page_afd = total_fixation_time / total_fixation_count
                sentence = sentence + f"Reader {i+1} : {round(user_page_afd,3)} ms\n"
            sentence = sentence + "\n"

            # roaft for all user for that paragraph
            users_roaft = []
            for i in range(len(self.log_folders)):
                users_roaft.append(self.para_informations[i]["roafts"][clicked_aoi])
            max_roaft = max(users_roaft)
            max_roaft_index = users_roaft.index(max_roaft)
            sentence = sentence + "Ratio of ON-target:All-target Fixation Time (ROAFT) : \n"
            for i in range(len(users_roaft)):
                sentence = sentence + f"Reader {i+1} ROAFT : {users_roaft[i]}\n"
            if max_roaft != 0:
                sentence = sentence + f"Reader {max_roaft_index+1} has the highest roaft\n\n"
            else:
                sentence = sentence + "\n\n"

            # AFD for all user for that paragraph
            users_afd = []
            for i in range(len(self.log_folders)):
                users_afd.append(self.para_informations[i]["average fixation duration aoi"][clicked_aoi])

            max_afd = max(users_afd)
            max_afd_index = users_afd.index(max_afd)

            sentence = sentence + "Average Fixation Duration : \n"

            for i in range(len(users_afd)):
                sentence = sentence + f"Reader {i+1} AFD : {users_afd[i]}\n"
            if max_afd != 0:
                sentence = sentence + f"Most time spend : Reader {max_afd_index+1} \n\n"
            else:
                sentence = sentence + "\n\n"


            # coverage for all user for that paragraph
            temp = []
            for user_id in range(len(self.log_folders)):
                coverage = self.para_informations[user_id]["coverages"][clicked_aoi]
                if  coverage != 0:
                    temp.append((user_id, coverage))
            if len(temp) > 0:
                sentence = sentence + "Coverages : \n"
                for i in range(len(temp)):
                    user_id, coverage = temp[i][0], temp[i][1]
                    sentence = sentence + f"Reader {user_id+1} : {coverage}\n"
                sentence = sentence + "\n"

            # forward saccades for all user for that paragraph if there is any
            temp = []
            highest_number = 0
            for user_id in range(len(self.log_folders)):
                forward = self.para_informations[user_id]["forward saccades per aoi"][clicked_aoi]
                if forward > highest_number:
                    highest_number = forward
                if  forward != 0:
                    temp.append((user_id, forward))
            if len(temp) > 0:
                sentence = sentence + "Forward saccades : \n"
                for i in range(len(temp)):
                    user_id, forward = temp[i][0], temp[i][1]
                    sentence = sentence + f"Reader {user_id+1} : {forward}\n"
                highest_user = []
                for i in range(len(temp)):
                    if temp[i][1] == highest_number:
                        highest_user.append(temp[i][0])
                users = ", ".join([str(i+1) for i in highest_user])
                if len(highest_user) < 4:
                    sentence = sentence + f"Reader {users} jump forward(s) more than others\n"
                if len(highest_user) == 4:
                    sentence = sentence + f"Reader {users} jump forward \n"
                sentence = sentence + "\n"

            # backward saccades for all user for that paragraph if there is any
            temp = []
            highest_number = 0
            for user_id in range(len(self.log_folders)):
                backward = self.para_informations[user_id]["backward saccades per aoi"][clicked_aoi]
                if backward > highest_number:
                    highest_number = backward
                if  backward != 0:
                    temp.append((user_id, backward))
            if len(temp) > 0:
                sentence = sentence + "Backward saccades : \n"
                for i in range(len(temp)):
                    user_id, backward = temp[i][0], temp[i][1]
                    sentence = sentence + f"Reader {user_id+1} : {backward}\n"
                highest_user = []
                for i in range(len(temp)):
                    if temp[i][1] == highest_number:
                        highest_user.append(temp[i][0])
                users = ", ".join([str(i+1) for i in highest_user])
                if len(highest_user) < 4:
                    sentence = sentence + f"Reader {users} revisit(s) this paragraph more than others\n"
                if len(highest_user) == 4:
                    sentence = sentence + f"Reader {users} revisit this paragraph\n"
                sentence = sentence + "\n"

            self.ids.report.text = sentence

    def get_clicked_aoi(self, x, y):
        # function to get the clicked paragraph or aoi based on the 
        # clicked pixels
        clicked_aoi = None
        for idx, data in enumerate(self.aois.items()):
            key, value = data
            aoi_x, aoi_y, aoi_w, aoi_h = value
            if aoi_x <= x <= aoi_x + aoi_w and aoi_y <= y <= aoi_y + aoi_h:
                clicked_aoi = idx
                break
        return clicked_aoi

    def get_aoi_heatmap(self, clicked):
        # function to generate heatmap when a paragraph or aoi is clicked 
        aoi_x, aoi_y, aoi_w, aoi_h = None, None, None, None 
        for idx, data in enumerate(self.aois.items()):
            key, value = data
            aoi_x, aoi_y, aoi_w, aoi_h = value
            if idx == clicked:
                break

        all_user_fixations = []
        for log_folder in self.log_folders:
            log_file = log_folder + "/" + f"{self.current_page+1}.log"
            fixations = get_fixations_from_gaze(log_file, self.display_size)
            all_user_fixations.extend(fixations)

        paragraph_fixations = []
        for fixation in all_user_fixations:
            _, _, _, x, y = fixation
            if aoi_x <= x <= aoi_x + aoi_w and aoi_y <= y <= aoi_y + aoi_h:
                paragraph_fixations.append(fixation)
        
        heatmap_matrix = get_heatmap(self.display_size, paragraph_fixations)

        img = Image.open(self.last_created_image)
        img = img.resize(self.display_size, Image.LANCZOS)

        plt.imshow(img)
        ax = plt.gca()

        ax.imshow(heatmap_matrix, cmap='jet', alpha=0.5)
        plt.axis('off')
        if os.path.exists(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap_new.png'):
            os.remove(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap_new.png')

        plt.savefig(f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap_new.png', 
                    dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close()

        self.ids.image1.source = f'./widgets/pdf_view_cache/visualizations/{self.current_page}_heatmap_new.png'
        self.ids.image1.reload()

        
    # function to detect click on a paragraph and showing 
    # informatio regarding that paragraph
    def get_pixel(self, instance, touch):
        if self.ids.image1.collide_point(*touch.pos):
            original_width, original_height = self.display_size
            ix, iy = self.ids.image1.norm_image_size
            scale_x, scale_y = original_width / ix, original_height / iy
            ox, oy = self.ids.image1.center_x - (ix / 2), self.ids.image1.center_y - (iy / 2)
            pixel_x, pixel_y = (touch.x - ox) * scale_x, (touch.y - oy) * scale_y
            pixel_x = max(0, min(pixel_x, original_width))
            pixel_y = max(0, min(pixel_y, original_height))
            pixel_y = self.display_size[1]-pixel_y
            clicked = self.get_clicked_aoi(pixel_x, pixel_y)
            if clicked != None:
                if self.ids.combined.state ==  "down" and self.fixation_clicked == None:
                    self.get_aoi_heatmap(clicked)
                    self.show_combined_info(clicked)


    def get_all_para_information(self, aois):
        # collect iformation of all aois for all users at the same time
        # save it in a list of dictionary and send it to readig_pattern_analysis function
        # where for all those data user specific threshold is generated to find out 
        # reading pattern
        analysis_report = []
        paragraph_informations = []
        for log_folder in self.log_folders:

            log_file = log_folder + "/" + f"{self.current_page+1}.log"
            fixations = get_fixations_from_gaze(log_file, self.display_size)
            fixation_times = calculate_fixation_time_AOIs(aois, fixations)
            fixation_counts = caluclate_fixation_count_AOIs(aois, fixations)
            coverage_aois = get_aoi_coverages(aois,fixations)
            average_fixation_duration_aois = caluclate_average_fixation_duration_AOIs(fixation_times, fixation_counts)
            roafts = calculate_roaft_aois(aois, fixation_times, total_fixation_duration = sum(fixation_times))
            saccades = calculate_saccades(fixations)
            backward_saccades = calculate_backward_saccades(saccades)
            forward_saccades = calculate_forward_saccades(saccades)
            forward_saccades_count_aoi = aoi_saccades_counts(aois, forward_saccades)
            backward_saccades_count_aoi = aoi_saccades_counts(aois, backward_saccades)
            saccades_count_aoi = aoi_saccades_counts(aois, saccades)
            saccades_lengths_aois = calculate_saccades_length_AOIs(aois, saccades)
            forward_saccades_length_aois = calculate_saccades_length_AOIs(aois, forward_saccades)
            backward_saccades_length_aois = calculate_saccades_length_AOIs(aois, backward_saccades)
            regression_rate_aois = calculate_progreg_rate_aoi(backward_saccades_count_aoi, saccades_count_aoi)
            progression_rate_aois = calculate_progreg_rate_aoi(forward_saccades_count_aoi, saccades_count_aoi)

            paragraph_information = {
                "fixation times per aoi" : fixation_times,
                "fixation counts per aoi" : fixation_counts,
                "roafts" : roafts,
                "coverages" : coverage_aois,
                "average fixation duration aoi" : average_fixation_duration_aois,
                "forward saccades per aoi" : forward_saccades_count_aoi,
                "backward saccades per aoi" : backward_saccades_count_aoi,
                "saccades per aoi" : saccades_count_aoi,
                "saccades lengths per aoi" : saccades_lengths_aois,
                "forward saccades lengths per aoi" : forward_saccades_length_aois,
                "backward saccades lengths per aoi" : backward_saccades_length_aois,
                "regression rates per aoi": regression_rate_aois,
                "progression rates per aoi": progression_rate_aois
            }
            paragraph_informations.append(paragraph_information)
            # get report for the current user
            analysis_report_per_user = readig_pattern_analysis(aois, paragraph_information )
            # save it in a list
            analysis_report.append(analysis_report_per_user)


        # get how many users skimmed, throughly read or found it confusing
        user_reading_pattern ={}
        for i in range(len(aois)):
            user_reading_pattern[i] = [0,0,0,0]
        for analysis in analysis_report:
            for aoi_index, dictionary in analysis.items():
                if dictionary["Reading Pattern"] == "Skimmed":
                    user_reading_pattern[aoi_index][0] = user_reading_pattern[aoi_index][0] + 1
                elif dictionary["Reading Pattern"] == "Read Thoroughly":
                    user_reading_pattern[aoi_index][1] = user_reading_pattern[aoi_index][1] + 1
                elif dictionary["Reading Pattern"] == "Skipped":
                    user_reading_pattern[aoi_index][2] = user_reading_pattern[aoi_index][2] + 1
                elif dictionary["Reading Pattern"] == "Undefined":
                    user_reading_pattern[aoi_index][3] = user_reading_pattern[aoi_index][3] + 1

        return paragraph_informations, user_reading_pattern, analysis_report