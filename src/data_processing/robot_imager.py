
import sys
import os

sys.path.append('..')
from utils.object_information import ObjectInformation, ObjectAttributes, ExperimentAttributes
from utils import utilities
from utils import generic_loader 
from utils import generic_loader 
from utils import visualization
import time
from image_processor import *

import argparse


parser = argparse.ArgumentParser(description="A script that process robot collected data to create mmWave image")
parser.add_argument("--name", type=str, default="" , help="Object Name. Only one of name or ID is required")
parser.add_argument("--id", type=str, default="", help="Object ID #. Only one of name or ID is required")
parser.add_argument("--ext", type=str, default="", help="Extension to save with")
parser.add_argument("--radar_type", type=str, default="77_ghz", help="the 77_ghz or 24_ghz radar?")
parser.add_argument("--is_los", type=str, default="y", help="Is LOS? (y/n)")
parser.add_argument("--bg_date", type=str, default="000", help="What day was environmnent background collected?")
parser.add_argument("--exp_num", type=str, default="1", help="What experiment number to process?")

args = parser.parse_args()
obj_name = args.name
obj_id = args.id
obj_angles = args.angles
ext = args.ext
radar_type = args.radar_type
exp_num = args.exp_num
is_los = True if (args.is_los == 'y' or args.is_los == 'Y') else False

assert radar_type in ['77_ghz', '24_ghz'], "Please choose a valid radar type."

# Fill in missing info
obj_info = ObjectInformation()
assert not (obj_name == '' and obj_id == ''), "Both name and ID can't be empty"
obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)

t1 = time.time()
# Set up loader and visualizer
loader = generic_loader.GenericLoader(obj_id, obj_name, is_sim=False, mult_files=True, is_los=is_los, exp_num=exp_num)
processor = ImageProcessor(False)
viz = visualization.Visualizer()

print(f'Running {radar_type} robot imager for {obj_id}_{obj_name} (LOS: {is_los}) ')

# Processes the image
robot_data, radar_data, robot_loc_data, exp_data = loader.load_all_data(radar_type)

# Correlate location and measurements
data = processor.correlate_locs_and_meas(radar_type, robot_data, radar_data, exp_data, robot_loc_data, speed="speed_8")

bg_chirp_num = None
bg_data=None
if radar_type == "24_ghz":
    # Load 24GHz radar calibration data
    if obj_name == 'EMPTY':
        if int(obj_name) > '0210':
            radar_calibration_id = '0213'
        else:
            radar_calibration_id = '000'
    else:
        radar_calibration_id = obj_info.get_object_info(ExperimentAttributes.CALIBRATION_ID_24, obj_name, obj_id, exp_num=exp_num)
    loader_bg = generic_loader.GenericLoader(obj_id=radar_calibration_id, name="background", is_sim=False, is_los=True, exp_num='1')
    robot_data_bg, radar_data_bg, robot_loc_data_bg, exp_data_bg = loader_bg.load_all_data("24_ghz")
    bg_data = processor.correlate_locs_and_meas("24_ghz", robot_data_bg, radar_data_bg, exp_data_bg, robot_loc_data_bg, speed="speed_8")

# Generate sar image
image, locs, robot_locs = processor.generate_sar_image_radar_frame(radar_type, data, exp_data, data_bg=bg_data, plot=False)
loader.save_image(radar_type, image, *locs, antenna_locs=robot_locs, ext=ext)
t2 = time.time()
print(f'Took {(t2-t1)} sec to finish processing image')

