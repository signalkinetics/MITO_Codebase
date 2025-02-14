"""" This file contains code for visualizing results. 
You can run this file to visualize a single image, or import it to access the visualization functions.  """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import copy
import sys

sys.path.append('..')
from utils import utilities
from utils.object_information import ObjectInformation, ObjectAttributes
from utils.generic_loader import *

class Visualizer:
    """
    Visualizer includes helper functions to visualize 3D images or 1D range profiles
    """

    def plot_sar_image(self, image, x_locs, y_locs, z_locs, plot_dim=2, normalization=None, plot_all_slices=False, obj_name='', title=None):
        """
        Plot Synthetic Aperture Radar (SAR) image 

        Parameters:
            image (numpy array): image to plot. Size: (L,W,H)
            x_locs (np array): x coordinates for each voxel in the image. Size: (L,)
            y_locs (np array): y coordinates for each voxel in the image. Size: (W,)
            z_locs (np array): z coordinates for each voxel in the image. Size: (H,)
            plot_dim (bool): the axis to project the data over. 0 = x, 1 = y, 2 = z (default: 2). 
                        E.g., plot_dim=2 would be an overhead view while 0 or 1 are side views
            normalization (matlabe Normalize): Normalization to apply for plotting or None to use default
            plot_all_slices (bool): Plot all individual image slices before plotting average image (along plot dim). Default: False
            obj_name (str): Object name to include in title of plot or an empty string to ignore
        Returns: None
        """
        # Compute magnitude of image
        abs_image = copy.deepcopy(np.abs(image))

        title_start = 'SAR Image' if obj_name == '' else f'SAR Image of {obj_name}'
        if plot_dim == 0: iter_locs = x_locs; axis_1_locs = y_locs; axis_2_locs = z_locs; axis_1_label='Y'; axis_2_label = 'Z'
        if plot_dim == 1: iter_locs = y_locs; axis_1_locs = x_locs; axis_2_locs = z_locs; axis_1_label='X'; axis_2_label = 'Z'
        if plot_dim == 2: iter_locs = z_locs; axis_1_locs = x_locs; axis_2_locs = y_locs; axis_1_label='X'; axis_2_label = 'Y'

        # Plot individual image slices
        if plot_all_slices:
            # Create normalization if not provided
            if normalization is None:
                normalization = Normalize(vmin=0, vmax=np.max(abs_image[:,:], axis=None))

            # Plot all slices
            for iter_loc in range(len(iter_locs)):
                if plot_dim == 0: image_slice = abs_image[iter_loc, :, :]; title=f'{title_start}: X = {x_locs[iter_loc]}'
                if plot_dim == 1: image_slice = abs_image[:, iter_loc, :]; title=f'{title_start}: Y = {y_locs[iter_loc]}'
                if plot_dim == 2: image_slice = abs_image[:, :, iter_loc]; title=f'{title_start}: Z = {z_locs[iter_loc]}'
                
                plt.pcolormesh(axis_1_locs, axis_2_locs, image_slice.T, norm=normalization, cmap = 'jet')
                plt.colorbar()
                plt.title(title)
                plt.xlabel(axis_1_label)
                plt.ylabel(axis_2_label)
                plt.axis('equal')
                plt.show()

        # Average the image along the plotting dimension
        avg_image = np.sum(abs_image[:,:,:], axis=plot_dim) / abs_image.shape[plot_dim]
        # avg_image = avg_image**2

        # Create normalization if not provided
        if normalization is None:
            normalization = Normalize(vmin=0, vmax=np.max(avg_image, axis=None)*0.5)

        # Plot average image
        if title is None:
            if plot_dim == 0: title=f'{title_start}: Avg along X'
            if plot_dim == 1: title=f'{title_start}: Avg along Y'
            if plot_dim == 2: title=f'{title_start}: Avg along Z'
        plt.pcolormesh(axis_1_locs, axis_2_locs, avg_image.T, norm=normalization, cmap = 'jet',linewidth=0,rasterized=True)
        plt.colorbar()
        plt.title(title)
        plt.xlabel(axis_1_label)
        plt.ylabel(axis_2_label)
        plt.axis('equal')
        # title = int(int(title.split('_')[-1])/4)
        # plt.savefig(f'/home/ldodds/clip_ap2/{title}.png')
        plt.show()
        
    def plot_range_profile(self, radar_type, is_sim, data, index, data_bg=None, max_distance=0.5):
        """
        Plot the 1D range profile of the radar data for the given measurement index.
        Compares the data before background subtraction and after subtraction if data_bg is given.

        Parameters:
            radar_type (str): type of radar to use ("24_ghz, 77_ghz")
            is_sim (bool): True if simulation data. False if real-world data
            data (np array): numpy array data containing the radar data
            index (int): the index of the location where we want to perform the FFT.
            data_bg (np array): numpy array background data for subtraction in 24_ghz radars. If None, doesn't perform background subtraction.
            max_distance (float): Plot the range profile from 0m until this max distance (in meters). Default: 0.5m
        """
        # Number of points to compute the FFT over
        # Increase this number to further interpolate the FFT. Decrease to improve computational complexity
        fft_size = 8192 

        radar_params = utilities.get_radar_params(is_sim=is_sim, radar_type=radar_type)
        bandwidth = radar_params['bandwidth']
        num_samples = radar_params['num_samples']

        # Load radar data
        radar_data = np.array(data["radar_data"])
        data = copy.deepcopy(radar_data[index])

        # Zero-Pad data (will increase resolution of FFT)
        data_padded = np.zeros((fft_size,), dtype=np.complex128)
        data_padded[:len(data)] = data[:,0]

        # Use FFT to compute time domain signal
        time_domain_signal = np.fft.fft(data_padded)

        # Load background data if applicable
        if data_bg is not None:
            radar_data_bg = np.array(data_bg["radar_data"]) 
            data_bg = copy.deepcopy(radar_data_bg[index])

            # Background subtraction
            data_bg_sub = data_padded[:len(data)] - data_bg
            time_domain_signal_subtracted = np.fft.fft(data_bg_sub)

        # Compute distances for given FFT size
        distances = np.arange(0, fft_size) * 3e8 / (2 * bandwidth) * num_samples / fft_size

        # Find index corresponding to maximum distance
        max_dist_ind = len(distances[distances<max_distance]) 

        # Plot figures
        plt.figure()
        if data_bg is None:
            plt.plot(distances[:max_dist_ind], np.absolute(time_domain_signal[:max_dist_ind]))
        else:
            plt.plot(distances[:max_dist_ind], np.absolute(time_domain_signal[:max_dist_ind]), label="Before Subtraction")
            plt.plot(distances[:max_dist_ind], np.absolute(time_domain_signal_subtracted[:max_dist_ind]), label= "After Subtraction")
            plt.legend()
        plt.xlabel("Range (m)")
        plt.ylabel("Amplitude")
        plt.title(f"Range Profile")
        plt.grid()
        plt.show()
        plt.close("all")

def load_and_plot(obj_name, obj_id, is_sim, is_los, exp_num, angles, is_specular_sim, is_edge_sim, radar_type):
    """
    Load and plot a single object
    Parameters:
        obj_name: Name of object
        obj_id: ID of object (as string)
        is_sim: Plot the simulation data?
        is_los: Plot the LOS data? (Ignored if is_sm is true)
        exp_num: Which experiment number to plot
        angles: Which angle to plot
        is_specular_sim: Whether to plot the specular simulation
        is_edge_sim: Whether to plot the edge simulation
        radar_type: 77_ghz or 24_ghz
    """
    if is_sim:
        if is_specular_sim:
            image_file_ext = utilities.load_param_json()['processing']['specular_extension']
        elif is_edge_sim:
            image_file_ext = utilities.load_param_json()['processing']['edges_extension']
        else:
            image_file_ext = utilities.load_param_json()['processing']['diffuse_extension']
    else:
        image_file_ext = utilities.load_param_json()['processing']['robot_collected_extension']

    # Get object ID
    obj_info = ObjectInformation()
    obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)
    
    # Load image data
    loader = GenericLoader(obj_id, obj_name, is_sim=is_sim, is_los=is_los, exp_num=exp_num)
    crop = obj_info.get_object_info(ExperimentAttributes.CROP, obj_id=obj_id, name=obj_name, exp_num=exp_num)
    crop_high = obj_info.get_object_info(ExperimentAttributes.CROP_HIGH, obj_id=obj_id, name=obj_name, exp_num=exp_num)
    attr = ExperimentAttributes.LOS_BACKGROUND_ID if is_los else ExperimentAttributes.NLOS_BACKGROUND_ID
    background_exp = obj_info.get_object_info(attr, obj_id=obj_id, name=obj_name, exp_num=exp_num) if radar_type=='24_ghz' and not is_sim else None
    image, (x_locs, y_locs, z_locs), antenna_locs = loader.load_image_file(radar_type, angles=angles, background_subtraction=background_exp, ext=image_file_ext, crop=crop, crop_high=crop_high)

    # Transpose image (for consistency)
    image = image.transpose([1,0,2])
    tmp = y_locs
    y_locs = x_locs
    x_locs = tmp

    # Plot Image
    visualizer = Visualizer()
    visualizer.plot_sar_image(image, x_locs, y_locs, z_locs, plot_dim=2, normalization=None, plot_all_slices=False, obj_name=obj_name, title=image_file_ext[1:])

def load_and_plot_rgb(obj_name, obj_id, exp_num):
    """
    Loads and plots a single RGB image (from a real-world experiment). This will apply the ground-truth segmentation mask to the RGB image
    Parameters:
        obj_name: Name of object
        obj_id: ID of object (as string)
        exp_num: Which experiment number to plot
    """
    # Load image and segmentation mask
    obj_info = ObjectInformation()
    obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)
    los_loader = GenericLoader(obj_id=obj_id, name=obj_name, is_sim=False, is_los=True, exp_num=exp_num) # For camera GT
    mask_data = los_loader.load_camera_masks()
    masked_depth = mask_data['masked_depth']
    cam_data = los_loader.load_camera_data()
    rgb = cam_data['rgb'].astype(np.uint8)
    mask = mask_data['rgbd_mask']
    mask = mask[0]

    # Apply segmentation mask
    rgb[:,:,0][np.logical_not(mask)] = 255
    rgb[:,:,1][np.logical_not(mask)] = 255
    rgb[:,:,2][np.logical_not(mask)] = 255
    rgb = np.flip(rgb, 1)

    # Plot image
    plt.imshow(rgb)
    plt.show()

if __name__=='__main__':
    """ This will plot a single image """
    is_sim = False # Whether to plot simulation or real data
    is_edge_sim = False # Whether to plot edge sim (doesn't matter if plotting real data)
    is_specular_sim = False # Whether to plot specular sim (doesn't matter if plotting real data)
    is_los = True # Whether to plot LOS or NLOS (doesn't matter if simulation)
    exp_num = 1 # Which exp number to load
    angles = None # Object angle to load

    obj_name = "padlock" # Name of object to plot
    obj_id = None # Will be filled in later

    if not is_sim: load_and_plot_rgb(obj_name, obj_id, exp_num) # RGB
    load_and_plot(obj_name, obj_id, is_sim, True, exp_num, angles, is_specular_sim, is_edge_sim, '77_ghz') # 77 LOS
    if not is_sim: load_and_plot(obj_name, obj_id, is_sim, False, exp_num, angles, is_specular_sim, is_edge_sim, '77_ghz') # 77 NLOS
    load_and_plot(obj_name, obj_id, is_sim, True, exp_num, angles, is_specular_sim, is_edge_sim, '24_ghz') # 24 LOS
    if not is_sim: load_and_plot(obj_name, obj_id, is_sim, False, exp_num, angles, is_specular_sim, is_edge_sim, '24_ghz') # 24 NLOS

