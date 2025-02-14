"""
This file is used to evaluate the performance of the segmentation algorithm 
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import torch
import os
import time

sys.path.append('..')
import segmentation
from utils.generic_loader import *
from utils import utilities

def _transform_point_cloud(points, transformation):
    """
    Transforms a point cloud from the camera's frame to the world frame.

    Args:
        points (np.ndarray): N x 3 array of points in the camera frame. - camera XYZ
        transformation (np.ndarray): 4x4 transformation matrix representing the camera pose in the world frame.

    Returns:
        np.ndarray: N x 3 array of points transformed to the world frame. - base link
    """
    # Convert points to homogeneous coordinates (N x 4 matrix)
    num_points = points.shape[0]
    points_homogeneous = np.ones((num_points, 4))
    points_homogeneous[:, :3] = points[:,:3]  # Set x, y, z coordinates

    # Apply the camera pose transformation to each point
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation.T)

    # Convert back to non-homogeneous coordinates by discarding the last column
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points


class SegmentationEvaluation:
    """
    This class evaluates the segmentation performance for a single object
    """
    def __init__(self, obj_id, name, radar_type, ext='', is_sim=False, is_los=True, radar_segmenter=None, exp_num='1', crop=True, crop_high=False):
        """
        Parameters:
            - obj_id (str): ID number of object to segment
            - name (str): Name of object to segment
            - radar_type (str): Radar type ('77_ghz' or '24_ghz')
            - x/y/z_angle (float): X/Y/Z angles to load
            - ext (str): Extension of the image to load
            - is_sim (bool): Whether to load the simulation or robot_collected image. Default: False
            - is_los (bool): whether to load the line-of-sight image. Default: True
            - radar_segmentater (Segmentation object): Segmentation object to use. If None, a new object will be created. Defualt: None
        """
        self.obj_id = obj_id
        self.is_sim = is_sim
        self.radar_type = radar_type
        self.is_los = is_los
        self.obj_name = name

        # Load radar data
        self.loader = GenericLoader(obj_id, name, is_sim=is_sim, is_los=is_los, exp_num=exp_num)
        self.mmwave_image, locs, radar_poses = self.loader.load_image_file(radar_type=radar_type, background_subtraction=None, ext=ext, crop=crop, crop_high=crop_high)
        self.radar_xlocs = locs[0]
        self.radar_ylocs = locs[1]
        self.radar_zlocs = locs[2]
        if exp_num == '1': # Load radar poses for older experiments which dont save them in the image file
            radar_poses = radar_poses[()]
            if radar_poses is None:
                radar_poses = self.loader.load_robot_loc_files(radar_type)
            if type(radar_poses) == dict:
                radar_poses = radar_poses['tx_77_locs']

        # Create new segmentation object if necessary
        if radar_segmenter is None:
            self.radar_segmenter = segmentation.Segmentation()
        else: 
            self.radar_segmenter = radar_segmenter

        # Load camera-based masks for ground-truth
        self.los_loader = GenericLoader(obj_id, name, is_sim=is_sim, exp_num=exp_num, is_los=True)
        self.cam_masks = self.los_loader.load_camera_masks(ext='')
        self.masked_depth = self.cam_masks['masked_depth']
        self.full_mask = self.cam_masks['rgbd_mask'][0]

        # Sometimes the depth image is unable to capture the full object. We fill these points with the average depth of the object 
        missing_mask = np.zeros_like(self.full_mask)
        missing_mask[np.logical_and(np.isnan(self.masked_depth), self.full_mask==1)] = 1
        mean_depth = np.nanmean(self.masked_depth)
        self.masked_depth[missing_mask] = mean_depth

        # Convert the depth image to XYZ points
        self.cam_xyz = utilities.convert_depth_to_xyz(self.masked_depth)
        self.exp_num = exp_num
        if exp_num == '1':
            # Define offsets between radar/camera
            self.cam_pose = np.reshape(np.array(self.cam_masks['pose']), (4,4))
            cam_2_tool0_trans = np.array([0.017, 0.086, -0.054])
            cam_frame_rot = R.from_matrix(self.cam_pose[:3,:3])
            self.cam_location = self.cam_pose[:3, 3] + cam_frame_rot.apply(cam_2_tool0_trans)
            self.tool0_to_cam = np.array([0.06,0.015,-0.0])

            # Get radar locations
            self.radar_location = np.array([ 0.42,  -0.23,  0.23]) # Radar starting location is always the same
            radar_frame_rot = R.from_quat(radar_poses[0][-4:])
            radar_translation = np.array([-0.186, -0.068, 0.046])
            self.radar_location += radar_frame_rot.apply(radar_translation)

        # Convert camera mask to radar frame
        self._get_cam_mask_in_radar_frame()
       
    def transform_camera_to_radar(self, exp_num):
        """
        Convert camera XYZ points from camera frame to radar frame. 
        Note: Since the physical camera/radar mount changed between experiment 1/2, the transformation between the two is different depending on the experiment number.
        
        Parameters:
            exp_num (string): String of experiment number
        Returns:
            XYZ points of the camera's mask in the radar's frame
        """

        if exp_num == '1':
            # Get offset between radar frame and camera frame
            translation = self.radar_location - (self.cam_location - self.tool0_to_cam)

            # Flip coordinates to match
            cam_xyz_radar_frame = self.cam_xyz[:,:,:]
            cam_xyz_radar_frame[self.cam_xyz[:,:] == [0,0,0]] = np.nan
            cam_xyz_radar_frame = np.transpose(self.cam_xyz, (1,0,2))
            cam_xyz_radar_frame = cam_xyz_radar_frame[:,:,(1,0,2)]
            cam_xyz_radar_frame = np.flip(cam_xyz_radar_frame, 1)
            cam_xyz_radar_frame[:,:,0] *= -1
            cam_xyz_radar_frame[:,:,1] *= -1
            cam_xyz_radar_frame[:,:,2] *= -1

            # Apply rotation/translation
            camera_rot = R.from_euler('xyz', [0,0,-3], degrees=True)
            for i in range(cam_xyz_radar_frame.shape[0]):
                for j in range(cam_xyz_radar_frame.shape[1]):
                    cam_xyz_radar_frame[i,j] = camera_rot.apply(cam_xyz_radar_frame[i,j])
            cam_xyz_radar_frame = cam_xyz_radar_frame - translation 
            return cam_xyz_radar_frame

        if exp_num == '2':
            # self.cam_xyz shape in (480,640,3), the camera points should be (N,3) where max N is 480*640
            a = self.cam_xyz.reshape(-1, 3)
            camera_points= a 

            # Pose of start of radar image
            radar_pose = [[1.0,  0.0,  0.0,  0.42],
                        [0.0,  1.0,  0.0, -0.23],
                        [0.0,  0.0,  1.0,  0.23],
                        [0.0,  0.0,  0.0,  1.0]]
            radar_pose = np.array(radar_pose).reshape(4,4)
            reverse_radar_transformation =  np.linalg.inv(radar_pose)

            # Pose where camera measurement is taken
            camera_pose = [ 0.99948083, -0.00142446, -0.03218756,  0.58516503, -0.0013166,  -0.99999345, 0.00337192, -0.00280226, -0.03219215, -0.00332779, -0.99947616,  0.4604748, 0.,  0., 0., 1. ]
            camera_pose = np.array(camera_pose).reshape(4,4)
            camera_pose[1,3] = camera_pose[1,3] + 0.01 
            camera_pose[0,3] = camera_pose[0,3] - 0.005 

            # Convert camera points to radar
            camera_points_in_world = _transform_point_cloud(camera_points, camera_pose)
            camera_points_in_radar = _transform_point_cloud(camera_points_in_world, reverse_radar_transformation)
            camera_points_in_radar = camera_points_in_radar.reshape(self.cam_xyz.shape)
            return camera_points_in_radar

    def evaluate_segmentation(self, save_ext=None, plot=False):
        """
        Evaluate the segmentation of this object. 
        This function will run the segmentation and compute various metrics (e.g., IoU, FScore, Precision, Recall)
        Parameters:
            - save_ext (str): Extension to use when saving masks. Pass None to skip saving the masks. Default: None
            - plot (bool): Whether to plot debugging plots
        """
        # Generate segmentation and save masks (if applicable)
        self.radar_mask = self.radar_segmenter.segment(self.mmwave_image, self.is_los, plot=plot)
        if self.radar_mask is None:
            self.radar_mask = self.radar_segmenter.segment(self.mmwave_image, is_los=self.is_los, plot=False, use_high_threshold=True)
        if save_ext is not None:
            self.loader.save_radar_masks(self.radar_mask, self.radar_type, True, ext=save_ext, angles=None)

        # Computer IoU
        cam_mask = self.cam_mask.astype(bool)
        intersection = np.sum(np.logical_and(self.radar_mask, cam_mask))
        union = np.sum(np.logical_or(self.radar_mask, cam_mask))
        iou = intersection / union

        # Compute FScore / Precision / Recall
        true_positives = np.sum(np.logical_and(self.radar_mask,  cam_mask))
        true_negatives = np.sum(np.logical_and(~self.radar_mask,  ~cam_mask))
        false_positives = np.sum(np.logical_and(self.radar_mask,  ~cam_mask))
        false_negatives = np.sum(np.logical_and(~self.radar_mask,  cam_mask))
        fscore = true_positives / (true_positives + 0.5*(false_positives+false_negatives))
        precision = true_positives / (true_positives+false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print(f'FScore: {fscore}')

        # Debugging plots
        if plot:
            # Will plot 3 subplots. 
            # The first will be the radar mask (in blue). 
            # The second will be the camera mask (in orange) 
            # The third will be the two masks plotted on top of each other (Overlap will be red. Unique portions will be in their original color)
            normalize = Normalize(vmin=0, vmax=4)
            plt.subplot(1, 3, 1)
            plt.pcolormesh(self.radar_xlocs, self.radar_ylocs, self.radar_mask.T,norm=normalize, cmap = 'jet')
            plt.colorbar()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.subplot(1, 3, 2)
            plt.pcolormesh(self.radar_xlocs, self.radar_ylocs, (self.cam_mask*3).T, norm=normalize,cmap = 'jet')
            plt.colorbar()
            plt.title(f'IoU: {iou}. Fscore: {fscore}. Obj: {self.obj_name}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.subplot(1, 3, 3)
            mask_sum = self.radar_mask + cam_mask*3
            plt.pcolormesh(self.radar_xlocs, self.radar_ylocs, mask_sum.T, norm=normalize,cmap = 'jet')
            plt.colorbar()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.show()
        return iou, fscore, precision, recall

    def _get_cam_mask_in_radar_frame(self, plot=False):
        """
        Creates the camera's mask in the radar frames. 
        To do so, we convert the mask's XYZ points from the camera frame to the radar frame. 
        Then, we project the converted XYZ points back to a 2D mask.
        Parameters:
            - plot: whether to plot for debugging. Default: False
        """
        # Get XYZ points of camera's mask in the radar frame
        # print("exp_num is",self.exp_num)
        cam_xyz_radar_frame = self.transform_camera_to_radar(self.exp_num)
        
        # Debugging plot
        if plot:
            plt.imshow(cam_xyz_radar_frame)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        # For each XYZ point, find its nearest coordinate in the XY mask
        x_step = self.radar_xlocs[1] - self.radar_xlocs[0]
        y_step = self.radar_ylocs[1] - self.radar_ylocs[0]
        cam_nearest_pixel = cam_xyz_radar_frame
        cam_nearest_pixel[:,:,0] -= self.radar_xlocs[0]
        cam_nearest_pixel[:,:,0] /= x_step
        cam_nearest_pixel[:,:,1] -= self.radar_ylocs[0]
        cam_nearest_pixel[:,:,1] /= y_step
        cam_nearest_pixel = np.round(cam_nearest_pixel)

        # Fill mask with all the (non-nan) nearest pixels
        nonzero_indices = cam_nearest_pixel[~np.any(np.isnan(cam_nearest_pixel),axis=2), 0:2].astype(np.int32)
        nonzero_indices = np.reshape(nonzero_indices, (-1, 2) )
        self.cam_mask = np.zeros((len(self.radar_xlocs), len(self.radar_ylocs)), dtype=np.uint8)
        self.cam_mask[nonzero_indices[:,0], nonzero_indices[:,1]] = 1

        # Debugging plot
        if plot:
            plt.imshow(self.cam_mask)
            plt.show()
            plt.pcolormesh(self.radar_xlocs, self.radar_ylocs, self.cam_mask.T, cmap = 'jet')
            plt.colorbar()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.show()


if __name__ == '__main__':
    # If True, this file will evaluate the full dataset. If False, it will plot an example for a single object
    eval_full_dataset = False

    if not eval_full_dataset:
        # This code can be used to evaluate a single object, and plot the result
        obj_id, name = '004', 'sugar_box'
        exp_num = '2' 
        seg_eval = SegmentationEvaluation(obj_id, name, '77_ghz', utilities.load_param_json()['processing']['robot_collected_extension'], exp_num=exp_num)
        seg_eval.evaluate_segmentation(plot=True)
    else:
        # This evaluates the full dataset
        print(f'Using GPU: {torch.cuda.is_available()}')
        radar_segmenter = segmentation.Segmentation(use_cpu=not torch.cuda.is_available())
        obj_info = ObjectInformation()
        all_objects = obj_info.list_all_objects()
        all_exp_nums = ['2'] 
        ext = '' # Extension to save with (if save_masks is true)
        radar_type = '77_ghz'
        NUM_MASKS_PER_OBJ = 3 # Number of masks per object to evaluate
        save_masks = False # Whether to save all generated masks
        save_results = True # Whether to save pkl file of resulting accuracy
        load_results = False # Whether to load results from pkl file instead of running from scratch
        all_precisions = {}
        all_recalls = {}
        all_fscores = {}

        # Evaluate segmentation on all objects in dataset
        if not load_results:
            for obj_key in all_objects:
                obj_id, obj_name = obj_key
                # Evalaute segmentation for all exp numbers
                for exp_num in all_exp_nums:
                    ext = utilities.load_param_json()['processing']['robot_collected_extension']
                    if exp_num not in all_precisions:
                        all_precisions[exp_num] = {'all': []}
                        all_recalls[exp_num] = {'all': []}
                        all_fscores[exp_num] = {'all': []}
                    for phase in ['los', 'nlos']:
                        if phase not in all_precisions[exp_num]:
                            all_precisions[exp_num][phase] = []
                            all_recalls[exp_num][phase] = []
                            all_fscores[exp_num][phase] = []

                        if phase == 'los':
                            is_sim = False
                            is_los = True
                            avail_attr = ExperimentAttributes.LOS_AVAIL
                        if phase == 'nlos':
                            is_sim = False
                            is_los = False
                            avail_attr = ExperimentAttributes.NLOS_AVAIL

                        # Check if this object is available
                        obj_available = obj_info.get_object_info(avail_attr, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                        if not obj_available or np.isnan(obj_available): continue                        

                        # Evaluate the segmentation for this object and save the precision & recall
                        crop = obj_info.get_object_info(ExperimentAttributes.CROP, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                        crop_high = obj_info.get_object_info(ExperimentAttributes.CROP_HIGH, obj_id=obj_id, name=obj_name, exp_num=exp_num)
                        print(f'Starting object {obj_id} {obj_name} Phase: {phase} Experiment Number: {exp_num}')
                        seg_eval = SegmentationEvaluation(obj_id, obj_name, radar_type=radar_type,ext=ext, is_sim=is_sim, is_los=is_los, radar_segmenter=radar_segmenter, exp_num=exp_num, crop=crop, crop_high=crop_high)
                        for i in range(NUM_MASKS_PER_OBJ):
                            save_ext = f'{ext}_segmentaiton_eval_{i}' if save_masks else None
                            _, fscore, precision, recall = seg_eval.evaluate_segmentation(save_ext=save_ext, plot=False)
                            all_precisions[exp_num]['all'].append(precision)
                            all_precisions[exp_num][phase].append(precision)
                            all_recalls[exp_num]['all'].append(recall)
                            all_recalls[exp_num][phase].append(recall)
                            all_fscores[exp_num]['all'].append(fscore)
                            all_fscores[exp_num][phase].append(fscore)

            if save_results:
                folder = 'results/'
                os.makedirs(os.path.dirname(folder), exist_ok=True)
                data = {'all_precisions': all_precisions, 'all_recalls': all_recalls, 'all_fscores': all_fscores}
                with open(f'{folder}/segmentation_accuracy_{int(time.time())}.pkl', 'wb') as f:
                    pickle.dump(data, f)
        else:
            filename = 'results/segmentation_accuracy_'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            all_precisions = data['all_precisions']
            all_recalls = data['all_recalls']
            all_fscores = data['all_fscores']



        # Plot results
        def get_percentiles(results, percentile):
            return np.array([np.nanpercentile(results['2']['all'], percentile),
                             np.nanpercentile(results['2']['los'], percentile), 
                             np.nanpercentile(results['2']['nlos'], percentile)])
        utilities.plot_dual_bar(np.array([0, 3,5]), 
                                get_percentiles(all_precisions, 25), 
                                get_percentiles(all_precisions, 50), 
                                get_percentiles(all_precisions, 75), 
                                get_percentiles(all_recalls, 25), 
                                get_percentiles(all_recalls, 50), 
                                get_percentiles(all_recalls, 75), 
                                xtick_labels=['Overall', 'Line-of-Sight', 'Fully\nOccluded'], bar_labels=['Precision', 'Recall'], line_labels=[], xlabel='', ylabel='Precision/Recall (%)', ylabel2='', title='')

