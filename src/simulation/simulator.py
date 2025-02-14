"""
This file contains the code for simulating the image of a single object. 
Use the run_simulation.sh script to run this simulation.
"""
import numpy as np
import time
from psbody.mesh.visibility import visibility_compute
from scipy.constants import c
from progressbar import ProgressBar

from mesh_utilities import *
from cpp.simulation import Simulation
from utils import utilities
from data_processing import image_processor
if utilities.load_param_json()['processing']['use_cuda']:
    from data_processing import py_image_gpu
else:
    from data_processing.cpp.imaging import Imaging


class Simulator:

    def __init__(self, mesh, radar_type, voxel_res=(.005, .005, .01), simulate_specularity=False, specularity_normal_threshold=3, simulate_edges=False, edge_threshold=30, use_large_ap=False):
        """
        Initializes Simulator for simulating the process of generating SAR images using a specified mesh file. 
        By default, this will apply diffuse simulation unless otherwise specified. 

        Parameters:
            mesh (trimesh object): mesh to run the simulation on
            radar_type (str): the type of radar to use ("24_ghz", "77_ghz")
            voxel_res (tuple(float, float, float)): the x/y/z resolution of the resulting SAR image
            simulate_specularity (bool): Whether or not to model only specular reflections. This will only take effect if simulate_edges is False. Default: False
            specularity_normal_threshold (float): Threshold in degrees between normal vector and signal incident angle which is considered a successfully recovered specular reflection. Default: 3
            simulate_edges (bool): Whether or not to model only reflections from the object's edges. Default: False
            use_large_ap (bool): Whether to simulate a large aperture (for larger objects). Default: False
        """
        self.radar_type = radar_type
        self.edge_threshold = edge_threshold
        self.simulate_edges = simulate_edges
        self.specularity_normal_threshold = specularity_normal_threshold*np.pi/180
        self.simulate_specularity = simulate_specularity

        # Find properties of mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces.astype(np.uint32)
        self.face_angles = mesh.face_adjacency_angles*180/np.pi
        self.normals = mesh.vertex_normals
        self.normals = self.normals / np.tile(np.linalg.norm(self.normals, axis=1)[:,np.newaxis], (1,3)) # Normalize normals
        # Find all faces which have a "sharp" (large) angle with their neighbor (e.g., are on an edge)
        sharp_angles = np.where(np.abs(self.face_angles) > edge_threshold)[0] 
        # Find all verticies on an edge face
        edge_verticies = mesh.face_adjacency_edges[sharp_angles] 
        # Construct mask which can be applied to verticies to find only verticies on an edge
        self.edge_verticies_mask = np.zeros((len(self.vertices)), dtype=np.int64)  
        self.edge_verticies_mask[edge_verticies.flatten()] = 1 

        # Load radar parameters
        radar_params = utilities.get_radar_parameters(radar_type=radar_type, is_sim=True)
        self.min_f = radar_params['min_f']
        self.max_f = radar_params['max_f']
        self.slope = radar_params['slope']
        self.bandwidth = radar_params['bandwidth']
        self.samples_per_chirp = radar_params['num_samples']

        # Create locations where radar will move to. For now, this is hard coded to match the real-world experiments
        self.radar_height = 0.2 # Same as real-world experiments
        step = c / self.min_f / 4
        self.num_antenna_per_dim = [230 if not use_large_ap else 270,int(np.floor(0.55/step))] # Same as real-world experiments
        self.initial_radar_loc = self._get_initial_radar_loc(step, self.num_antenna_per_dim)
        self.radar_locations = self._radar_movement()

        # Check object entirely within radar's trajectory
        self.bbox = MeshTransformer().find_bounding_box(mesh)
        x_lim = self.bbox[0] - self.initial_radar_loc[0]
        y_lim = self.bbox[1] - self.initial_radar_loc[1]
        z_lim = self.bbox[2] - self.initial_radar_loc[2]
        assert y_lim[1] < self.radar_locations[-1,1]-self.initial_radar_loc[1] and \
               y_lim[0] > self.radar_locations[0,1]-self.initial_radar_loc[1] and \
               x_lim[1] < self.radar_locations[-1,0]-self.initial_radar_loc[0] and \
               x_lim[0] > self.radar_locations[0,0]-self.initial_radar_loc[0], \
               "Object should be contained entirely within radar trajectory to provide a good result"

        # Create the coordinates of the image
        self._create_image_coordinates(x_lim, y_lim, z_lim, voxel_res)

        # Initialize C++ simulator
        self.Simulator = Simulation(
            np.array(self.vertices),
            [x_lim[0], x_lim[1]], [y_lim[0], y_lim[1]], [z_lim[0], z_lim[1]], 
            radar_type
        )

    def simulate(self):
        """
        Runs the simulation software written in C++ and returns the generated SAR Image. 
        The simulation follows three main steps:
        1) Find which vertices are visible from each radar location
        2) Simulate the RF channel from each radar location
        3) Combine the RF channels into a final SAR image (in the same way as we do with data collected in a real-world experiment)

        Parameters: 
        
        Returns:
            - image (np array of floats): 3D Simulated SAR image. Shape: (X,Y,Z) (X=num voxels in x dimension, Y=num voxels in y dim, Z=num voxels in z dim)
            - channels (np array of floats): The simulated channels used to produce the SAR image. Shape: (L,N) (L=num of radar locations, N=number of samples per chirp)
            - radar_shifted_locations (np array of floats): The radar locations used to produce the SAR image. These have been shifted by the initial_radar_loc. Shape: (L,3)
        """
        # STEP 1: Find which vertices are visible from each radar location
        all_visible_vert = np.zeros((self.radar_locations.shape[0], len(self.vertices)))
        for i in ProgressBar()(range(self.radar_locations.shape[0])):
            # Find visible vertices (e.g., not obstructed by other parts of mesh)
            vertex_visibility, _ = visibility_compute(v=self.vertices, f=self.faces, n=self.normals, cams=np.double(np.array(self.radar_locations[i].reshape((1, 3)))))

            # Apply other filters if applicable
            if self.simulate_edges:
                # If we are only simulating edges, apply the mask for verticies on object edges
                vertex_visibility = np.logical_and(vertex_visibility, self.edge_verticies_mask)
            elif self.simulate_specularity:
                # If we are only simulating specular reflections, find verices which produce a specular reflection back towards this radar location
                # Find angle between signal incident angle and each vector normal using the following equation: 
                # theta = arccos(dot(incident vector, normal)/||normal|| ||incident vector||)
                incident_vec = self.vertices-self.radar_locations[i] # Vector from radar location to each vertex. This is the signal incident vector
                normal_dot_incident = np.sum(self.normals * incident_vec, axis=1) 
                normal_dot_incident /= np.linalg.norm(self.normals,axis=1)
                normal_dot_incident /= np.linalg.norm(self.vertices-self.radar_locations[i],axis=1)
                theta = np.abs(np.arccos(normal_dot_incident))

                # Apply mask for specular vertices. Allow for both 0 or 180 degree normals
                vertex_visibility = np.logical_and(vertex_visibility, np.logical_or(theta < self.specularity_normal_threshold, \
                                                                                    np.abs(np.pi-theta) < self.specularity_normal_threshold ))

            all_visible_vert[i]= vertex_visibility

        # STEP 2: Simulate RF channel for this radar location. 
        # The resulting channel is the sum of reflections from this radar location to all visible vertices and back to the radar.
        for i in ProgressBar()(range(self.radar_locations.shape[0])):
            self.Simulator.simulate_rf_channel(np.nonzero(all_visible_vert[i])[0], self.radar_locations[i], self.initial_radar_loc)
        channels, radar_shifted_locations = self._get_channels_and_radar()
        radar_shifted_locations = np.array(radar_shifted_locations)
        channels = np.array(channels)

        # STEP 3: Compute the final SAR image. This is the same process we use when imaging with real-world data
        apply_ti_offset = False
        rx_offsets = np.array([[0,0,0.0]])
        use_interpolated_processing = utilities.load_param_json()['processing']['use_interpolated_processing']
        if use_interpolated_processing: # Process on GPU
            image_proc = image_processor.ImageProcessor(radar_type=self.radar_type)
            measurement_data = image_proc.interpolate_measurements(channels)
        else:
            measurement_data = channels
        if utilities.load_param_json()['processing']['use_cuda']:
            image = py_image_gpu.image_cuda(self.x_locs, self.y_locs, self.z_locs, radar_shifted_locations, measurement_data, rx_offsets, self.slope, c/self.min_f, self.bandwidth, self.samples_per_chirp, apply_ti_offset, use_interpolated_processing=use_interpolated_processing)
        else:
            is_sim = True
            fft_spacing = np.float32(3e8/(2*self.bandwidth)*self.samples_per_chirp/(measurement_data.shape[1]))
            imaging = Imaging(self.x_locs, self.y_locs, self.z_locs, self.radar_type, is_sim)
            image = np.array(imaging.image(radar_shifted_locations, measurement_data, rx_offsets, fft_spacing, apply_ti_offset, use_interpolated_processing))
        return image, channels, radar_shifted_locations

    def _get_channels_and_radar(self):
        """
        Get the channels and radar locatons from c++
        Returns:
            - channels (np array of floats): Simulated RF channels. Shape: (L,N)
            - radar_shifted_locations (np array of floats): Locations used in simulation (shifted by initial radar location). Shape: (L,3)
        """
        channels = self.Simulator.get_channels()
        radar_shifted_locations = self.Simulator.get_shifted_radar_locations()
        return channels, radar_shifted_locations

    def _radar_movement(self):
        """
        Creates locations where the radar moves to.
        Returns: 
            - Radar locations (np array of floats): Locations of radar (relative to starting point of (0,0,0)). Will need to add offset of these locations relative to object. Shape: (L,3)
        """
        assert len(self.num_antenna_per_dim)==2, "self.num_antenna_per_dim needs to be initialized to a 2 long list"

        # Create radar locations as a dense grid of lambda/4 spacing with self.num_antenna_per_dim antennas in the X and Y dimensions
        spacing = []
        step = c / self.min_f / 4
        for i in range(self.num_antenna_per_dim[0]):
            for j in range(self.num_antenna_per_dim[1]):
                element = [i * step, j * step, 0.0]  
                spacing.append(element)
        return np.array(spacing) + np.array(self.initial_radar_loc)

    def _create_image_coordinates(self, x_lim, y_lim, z_lim, voxel_res): 
        """
        Sets the x/y/z_locs to generate the SAR image. 
        Parameters:
            - x_lim: 2 long list of [min_x, max_x] - The bounds of image in the x dimension
            - y_lim: 2 long list of [min_y, max_y] - The bounds of image in the y dimension
            - z_lim: 2 long list of [min_z, max_z] - The bounds of image in the z dimension
            - voxel_res: 3 long list of [res_x, res_y, res_z] - The resolution of the image in each dimension
        Notes:
            - All values are in meters. 
            - Z is the up dimension
        """
        min_x, max_x = x_lim
        min_y, max_y = y_lim
        min_z, max_z = z_lim
        res_x, res_y, res_z = voxel_res

        # For now, hard code to match  real world processing
        radar_max_x = self.radar_locations[-1,0]-self.initial_radar_loc[0]
        if self.radar_type =='77_ghz':
            min_x = 0 #- 0.05
            max_x = radar_max_x #- 0.05
            min_y = 0.05 #radar_min_y - 0.0*radar_y_diff# radar_y_diff / 4
            max_y = 0.5 #radar_max_y + 0.0*radar_y_diff# radar_y_diff / 4
        else:
            min_x = 0 - 0.05
            max_x = radar_max_x + 0.05
            min_y = 0.05 #radar_min_y - 0.0*radar_y_diff# radar_y_diff / 4
            max_y = 0.65 #radar_max_y + 0.0*radar_y_diff# radar_y_diff / 4
        min_z, max_z = -0.37, -0.15
        res_x = 0.0025 
        res_y = 0.0025 
        res_z = 0.01

        self.x_locs = np.arange(min_x, max_x, res_x)
        self.y_locs = np.arange(min_y, max_y, res_y)
        self.z_locs = np.arange(min_z, max_z, res_z)

    def _get_initial_radar_loc(self, step, num_antennas_per_dim):
        """
        Returns the initial position of the radar in the simulation
        """
        return [-step*num_antennas_per_dim[0]/2, -step*num_antennas_per_dim[1]/2, self.radar_height]