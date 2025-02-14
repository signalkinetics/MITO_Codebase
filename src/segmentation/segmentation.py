"""
This class can be used to segment a mmWave image with the SAM network
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import math

from segment_anything import sam_model_registry, SamPredictor

sys.path.append('..')
from utils.generic_loader import *
from utils import utilities
from segmentation_utilities import *
from utils.visualization import *


class Segmentation:
    def __init__(self, use_cpu=False):
        # Set up SAM model
        model_type = "vit_h"
        weights_path = utilities.check_for_sam_weights() # Get path to weights (and download them if they don't exist)
        device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")
        print(f'Using device: {device}')
        sam = sam_model_registry[model_type](checkpoint=weights_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def segment(self, image, is_los, obj_name='', plot=False, use_high_threshold=False):
        """
        Segment the given mmWave image with the SAM model.
        Parameters:
            - image (np array): mmWave image to segment (Shape: LxWxH)
            - is_los (bool): Is this a line-of-sight image?
            - obj_name (str): Name of object. This is only used for the title of any plots
            - plot (bool): Whether to plot the segmentation results. Default: False
            - use_high_threshold (bool): Whether to use the high power threshold during prompt point selection. Default: False
        """

        # Convert 3D mmWave Image to a 2D RGB image
        mmwave_2d_image = np.sum(np.abs(image), axis=2) / image.shape[2] # Average along height dimension
        mmwave_2d_image /= np.max(mmwave_2d_image) # Normalize image so max value = 1
        cmap = plt.get_cmap('jet') 
        mmwave_color_image = cmap(mmwave_2d_image)[:,:,:3] # Apply color map to get an RGB image
        mmwave_color_image = (mmwave_color_image * 255).astype(np.uint8) 

        max_image = mmwave_2d_image.copy()
        min_image = mmwave_2d_image.copy()
        high_threshold = 0.4 if is_los else .4
        low_threshold = 0.02 if is_los else .05
        if use_high_threshold:
            high_threshold = 0.5
            low_threshold = 0.2
        # mmwave_2d_image[min_image<np.max(mmwave_2d_image)*low_threshold] = 0
        min_image[min_image>np.max(mmwave_2d_image)*low_threshold] = 0 
        max_image[max_image<np.max(mmwave_2d_image)*high_threshold] = 0 ## used to be np.max(avg_img) * threshold
        all_points = np.nonzero(max_image)
        all_points = np.vstack((all_points[1], all_points[0])).T
        remove_points = np.nonzero(min_image)
        remove_points = np.vstack((remove_points[1], remove_points[0])).T


        ## Thresholding based on pixel density/size
        num_pnts = math.ceil(len(all_points)*.005) #0.005
        # num_pnts = 15
        if use_high_threshold:
            num_pnts = 5

        all_points = all_points[np.random.choice(np.arange(len(all_points)), (num_pnts,))]
        input_label = np.ones((len(all_points)))
        if len(remove_points) == 0:
            remove_points = []
        else:
            remove_points = remove_points[np.random.choice(np.arange(len(remove_points)), (num_pnts,))]
            remove_labels = np.zeros((len(remove_points)))
            all_points = np.vstack((all_points, remove_points))
            input_label = np.append(input_label, remove_labels)


        # Predict masks with SAM network
        self.predictor.set_image(mmwave_color_image)
        masks, scores, _ = self.predictor.predict(
            point_coords=all_points,
            point_labels=input_label,
            multimask_output=True,
        )

        # Dont choose best mask if its more than 20% of the image
        best_mask = np.argmax(scores)
        if np.sum(masks[best_mask]) > masks.shape[1]*masks.shape[2]*0.2:
            best_mask = np.argpartition(scores, -2)[-2]
            if np.sum(masks[best_mask]) > masks.shape[1]*masks.shape[2]*0.2:
                best_mask = np.argpartition(scores, -3)[-3]
                if np.sum(masks[best_mask]) > masks.shape[1]*masks.shape[2]*0.2 and not use_high_threshold:
                    return None


        # Plot for debugging
        if plot:
            # Plots the final mask, 2D grayscale image, the mmwave color image, and the mask overlayed on the color image
            plt.subplot(1,4,1)
            show_mask(masks[best_mask], plt.gca()) 
            plt.title(f"Mask {best_mask+1}, Score: {scores[best_mask]:.3f}", fontsize=18)
            plt.axis('off')
            plt.subplot(1,4,2)
            plt.imshow(mmwave_2d_image, cmap='gray')
            plt.title(obj_name, fontsize=18)
            plt.axis('off')
            plt.subplot(1,4,3)
            plt.imshow(mmwave_color_image)
            plt.title(obj_name, fontsize=18)
            plt.axis('off')
            plt.subplot(1,4,4)
            plt.imshow(mmwave_color_image)
            show_points(all_points, input_label, plt.gca())
            show_mask(masks[best_mask], plt.gca())
            plt.title(f"Mask {best_mask+1}, Score: {scores[best_mask]:.3f}", fontsize=18)
            plt.show()
        return masks[best_mask]
  

if __name__=='__main__':
    # Running this file will create masks for all (or some) objects
    # To run segmentation on a specific subset of objects, either obj_names or obj_ids should be a list of strings. The other one can be set to None
    # To run segmentation on all objects, both obj_names and obj_ids should be None.
    obj_names = ["sugar_box"] # None or a list of names
    obj_ids = None # None or a list of IDs (as strings)
    save_mask = False # Whether to save 
    is_sim = False # Whether to segment simulation
    los_types = [True] # List of booleans for whether to run LOS and/or NLOS. True will run LOS, False will run NLOS
    exps_to_process = ['1'] # Which experiment numbers to process
    ext = utilities.load_param_json()['processing']['robot_collected_extension'] # Which image extension to process
    radar_type = '77_ghz' # 77_ghz or 24_ghz
    plot = True # Whether to plot each segmentation

    # Fill in all_objects list with either provided objects or all objects (if none are provided)
    obj_info = ObjectInformation()
    all_objects = []
    if obj_names is None and obj_ids is None:
        all_objects = obj_info.list_all_objects()
    else: 
        if obj_names is not None:
            for obj_name in obj_names:
                all_objects.append(obj_info.fill_in_identifier_sep(obj_name, None))
        else:
            for obj_id in obj_ids:
                all_objects.append(obj_info.fill_in_identifier_sep(None, obj_id))

    
    # Run segmentation on each object / exp number / phase
    radar_segmenter = Segmentation(use_cpu=not utilities.load_param_json()['processing']['use_cuda'])
    for obj_id, obj_name in all_objects:
        for exp_num in exps_to_process:
            for is_los in los_types:
                # Check image availability
                avail_attr = ExperimentAttributes.SIM_AVAIL if is_sim else (ExperimentAttributes.LOS_AVAIL if is_los else ExperimentAttributes.NLOS_AVAIL)
                obj_available = obj_info.get_object_info(avail_attr, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                if not obj_available: continue

                # Load image
                loader = GenericLoader(obj_id, obj_name, is_sim=is_sim, exp_num=exp_num, is_los=is_los)
                crop = obj_info.get_object_info(ExperimentAttributes.CROP, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                crop_high = obj_info.get_object_info(ExperimentAttributes.CROP_HIGH, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                mmwave_image, locs, radar_poses = loader.load_image_file(radar_type=radar_type, background_subtraction=None, ext=ext, crop=crop, crop_high=crop_high)
                
                # Segment radar. If it fails, repeat with high threshold
                radar_mask = radar_segmenter.segment(mmwave_image, is_los, plot=plot)
                if radar_mask is None:
                    radar_mask = radar_segmenter.segment(mmwave_image, is_los=is_los, plot=plot, use_high_threshold=True)
                    if radar_mask is None:
                        raise Exception('Mask is None!')

                if save_mask:
                    loader.save_radar_masks(radar_mask, radar_type, True, ext=ext, angles=None)
                print(f'Finished object {obj_id} {obj_name} exp num: {exp_num} LOS: {is_los}')
        print(f'Finished all for object {obj_id} {obj_name}')

