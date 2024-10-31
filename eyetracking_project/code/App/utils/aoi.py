import cv2
import pytesseract
from . import preprocess as preprocess

# detect the aois (paragraphs, lines, words) and their boundaries
# using pytessaract library . using the level argument of this function
# we can get different types of aoi for paragraph level = 3 for lines
# level = 4 , for words level = 5 
def detect_area(image_path, display_size, level, aoi_type):
    img = cv2.imread(image_path)
    img = cv2.resize(img, display_size)
    # Configure Tesseract to detect boundaries 
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

    # Find bounding boxes
    boxes = []
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if d['level'][i] == level: 
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[1])
    AOIs = {}
    for index, line in enumerate(boxes):
        key = f"{aoi_type}_AOI_{index+1}"
        AOIs[key] = line
    return AOIs 

# split the aoi (which is a rectangle) into a matrix of cells
# which are also rectangles
def split_aoi(aoi, row, col):
    x, y, width, height = aoi
    cell_width = width / row
    cell_height = height / col
    grids = []
    for i in range(row):
        for j in range(col):
            cell_x = x + j * cell_width
            cell_y = y + i * cell_height
            grids.append((cell_x, cell_y, cell_width, cell_height))
    return grids

# calculate the coverage for each of the aoi present in an image
def get_aoi_coverages(aois, fixations):
    coverage_aois = []
    for key, aoi in enumerate(aois.items()):
        _, value = aoi
        grids = split_aoi(value, 2, 3)
        total_grid_touched = 0
        for grid in grids:
            grid_touched = 0
            cell_x, cell_y, width, height = grid
            for fixation in fixations:
                _, _, duration, x, y = fixation
                if cell_x <= x <= cell_x + width and cell_y <= y <= cell_y + height:
                    grid_touched = 1
            if grid_touched == 1:
                total_grid_touched = total_grid_touched + 1
        coverage = total_grid_touched / len(grids)
        coverage_aois.append(coverage)
    return coverage_aois



