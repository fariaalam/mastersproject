import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from preprocess import (
    prepare_raw_data,
    event_detecting_smoothing,
    get_fixations
)

#####################################################
# functions to call for fixations, heatmap, opacity #
#####################################################

# Function to get fixations
def get_fixations_from_gaze(log_path, display_size):
    gaze_df = pd.read_csv(log_path, delimiter = ",", header=None, 
            names = ['timestamp_system','left_gaze_point_x', 'left_gaze_point_y',
                    "right_gaze_point_x", "right_gaze_point_y"])
    gaze_df.dropna(inplace=True)

    proccessed_csv_file_path = 'proccessed_gaze_data.csv'

    prepare_raw_data(gaze_df, proccessed_csv_file_path, display_size)
    event_detecting_smoothing(proccessed_csv_file_path)
        
    fixations = get_fixations(proccessed_csv_file_path)
    return fixations


# function to get heatmap
def get_heatmap(display_size, fixations):
    def gaussian(x, sx, y=None, sy=None):

        # square Gaussian if only x values are passed
        if y == None:
            y = x
        if sy == None:
            sy = sx
        # centers
        xo = x/2
        yo = y/2
        # matrix of zeros
        M = np.zeros([y,x],dtype=float)
        # gaussian matrix
        for i in range(x):
            for j in range(y):
                M[j,i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )

        return M

    def generate_heatmap_matrix(display_size, fixations):
        gwh = 200
        gsdwh = gwh/6
        gaus = gaussian(gwh,gsdwh)

        strt = gwh/2
        heatmapsize = int(display_size[1] + 2*strt), int(display_size[0] + 2*strt)
        heatmap = np.zeros(heatmapsize, dtype=float)
        # create heatmap
        for i in range(0,len(fixations[:])):
            # get x and y coordinates
            #x and y - indexes of heatmap array. must be integers
            x = strt + int(fixations[i][3]) - int(gwh/2)
            y = strt + int(fixations[i][4]) - int(gwh/2)
            # correct Gaussian size if either coordinate falls outside of
            # display boundaries
            if (not 0 < x < display_size[0]) or (not 0 < y < display_size[1]):
                hadj=[0,gwh];vadj=[0,gwh]
                if 0 > x:
                    hadj[0] = abs(x)
                    x = 0
                elif dispsize[0] < x:
                    hadj[1] = gwh - int(x-display_size[0])
                if 0 > y:
                    vadj[0] = abs(y)
                    y = 0
                elif dispsize[1] < y:
                    vadj[1] = gwh - int(y-display_size[1])
                # add adjusted Gaussian to the current heatmap
                try:
                    heatmap[int(y):int(y+vadj[1]),int(x):int(x+hadj[1])] += \
                        gaus[int(vadj[0]):int(vadj[1]),int(hadj[0]):int(hadj[1])] * fixations[i][2]
                except:
                    # fixation was probably outside of display
                    pass
            else:
                # add Gaussian to the current heatmap
                heatmap[int(y):int(y+gwh),int(x):int(x+gwh)] += gaus * fixations[i][2]
        # resize heatmap
        heatmap = heatmap[int(strt):int(display_size[1]+strt),int(strt):int(display_size[0]+strt)]
        # remove zeros
        #heatmap = np.clip(heatmap, None, 255)
        heatmap = heatmap / np.max(heatmap)
        lowbound = np.mean(heatmap[heatmap>0])
        heatmap[heatmap<lowbound] = np.nan
        return heatmap
        
    heatmap_matrix = generate_heatmap_matrix(display_size, fixations)
    return heatmap_matrix

# function to get opacitymap
def get_opacity_map(heatmap):
    heatmap = np.nan_to_num(heatmap, nan=0)
    inverted_heatmap = 1 - heatmap
    opacity_matrix = np.repeat(inverted_heatmap[:, :, np.newaxis], 3, axis=2)
    return opacity_matrix


if __name__ == "__main__":

    display_size = (1280, 990)
    log_path = "logfiles_article_rotated_rotated.pdf_2024-03-12_13-50-21/1.log"

    # get fixations every fixations is list of lists
    # each list contains 5 values, start_time, end_time, duration, x , y
    fixations = get_fixations_from_gaze(log_path, display_size)

    # get heatmap matrix of size (display_size [0], display_size[1])
    heatmap_matrix = get_heatmap(display_size, fixations)

    # get opacity matrix of size (display_size [0], display_size[1], 3) , here 3 for rgb
    opacity_matrix = get_opacity_map(heatmap_matrix)
    
    # read the image
    image_path = 'Image1.jpeg'
    img = Image.open(image_path)
    img = img.resize(display_size, Image.LANCZOS)

    plt.imshow(img)
    ax = plt.gca()

    ax.imshow(heatmap_matrix, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig("heatmap_plot_2.png", dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()

    image = Image.open('heatmap_plot_2.png')
    image.show()
    







