from . import preprocess as preprocess

def calculate_fixation_time_AOIs(aois, fixations):
    # calculate the fixation time for each of the aois
    aois_fixation_times = []
    for key, value in aois.items():
        aoi_x, aoi_y, aoi_w, aoi_h = value
        fixation_time = 0
        for element in fixations:
            _, _, duration, fix_x, fix_y = element
            if aoi_x <= fix_x <= aoi_x + aoi_w and aoi_y <= fix_y <= aoi_y + aoi_h:
                fixation_time = fixation_time + duration
        aois_fixation_times.append(fixation_time)
    return aois_fixation_times

def caluclate_fixation_count_AOIs(aois, fixations):
    # calculate the fixation count for each of the aois
    aois_fixation_counts = []
    for key, value in aois.items():
        aoi_x, aoi_y, aoi_w, aoi_h = value
        counts = 0
        for element in fixations:
            _, _, _, fix_x, fix_y = element
            if aoi_x <= fix_x <= aoi_x + aoi_w and aoi_y <= fix_y <= aoi_y + aoi_h:
                counts = counts + 1
        aois_fixation_counts.append(counts)
    return aois_fixation_counts

def calculate_roaft_aois(aois, fixation_times, total_fixation_duration):
    # calculate the roafts for each of the aois
    roafts = []
    for index, data in enumerate(aois.items()):
        key, value = data
        fixation_time = fixation_times[index]
        roaft = fixation_time / total_fixation_duration
        roaft = round(roaft, 3)
        roafts.append(roaft)
    return roafts

def calculate_progreg_rate_aoi(forback_saccades_count_aoi, saccades_count):
    # calculate progression rate or regression rate from forward or backward saccade 
    # count and total saccade count
    progreg_rate = []
    for idx, element in enumerate(forback_saccades_count_aoi):
        try:
            progreg_rate.append(element/saccades_count[idx])
        except Exception as ex:
            progreg_rate.append(0)
    return progreg_rate

def caluclate_average_fixation_duration_AOIs(times, counts):
    # calculate the average fixation duration 
    # using fixation times and counts for each of the aois
    avg_duration = []
    for i in range(len(times)):
        try:
            avg_duration.append(times[i]/counts[i])
        except Exception as ex:
            avg_duration.append(0)
    return avg_duration

def readig_pattern_analysis(paragraphs, paragraph_information):
    # detect the reading of each of the aois for a user 
    total_fixation_time = sum(paragraph_information["fixation times per aoi"])
    total_fixation_count = sum(paragraph_information["fixation counts per aoi"])
    # average fixation duration of the whole page for user
    user_page_afd = total_fixation_time / total_fixation_count

    analysis = {}

    for i in range(len(paragraphs)):
        analysis[i] = {"Reading Pattern": "Undefined", "Comments": ""}

        # condition for reading pattern Read Thoroughly
        if paragraph_information["average fixation duration aoi"][i] > user_page_afd \
            and paragraph_information["coverages"][i] >= 0.5:
            analysis[i] = {"Reading Pattern": "Read Thoroughly", "Comments": ""}
        
        # condition for reading pattern Skimmed
        elif paragraph_information["average fixation duration aoi"][i] < user_page_afd \
            and 0.5 > paragraph_information["coverages"][i] >= 0.3:
            analysis[i] = {"Reading Pattern": "Skimmed", "Comments": ""}

        # condition for reading pattern Skipped
        elif paragraph_information["fixation times per aoi"][i] == 0:
            analysis[i] = {"Reading Pattern": "Skipped", "Comments": ""}

    return analysis










