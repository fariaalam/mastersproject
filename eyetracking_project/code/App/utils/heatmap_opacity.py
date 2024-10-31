import numpy as np

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
                elif display_size[0] < x:
                    hadj[1] = gwh - int(x-display_size[0])
                if 0 > y:
                    vadj[0] = abs(y)
                    y = 0
                elif display_size[1] < y:
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