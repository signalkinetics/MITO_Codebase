"""
This script runs the simulation for a single object based on the input arguments. 
Use run_simulation.sh to easily run this for a list of objects.
"""
import trimesh
import math
import time
import argparse
import sys

sys.path.append('..')
from utils.object_information import ObjectInformation, ObjectAttributes
from utils.generic_loader import *
from simulator import *
from mesh_utilities import *
from utils.visualization import *
from utils import utilities


def sim(simulate_specularity, specularity_normal_threshold, ext, noise,rotated_mesh,angles=(0,0,0), simulate_edges=False, edge_threshold=0, large_ap=False):
    """
    Run the simulator and save the output
    Parameters:
        - simulate_specularity (bool): If True, only simulate specular reflections. 
        - specularity_normal_threshold (float): Minimum angle in degrees which is considered a successfully recovered specular reflection
        - ext (str): Additional extension to add to end of filename
        - simulate_edges (bool): If True, only simulate reflections from edges
        - edge_threshold (float): Minimum angle in degrees between two faces to consider the veritices part of an edge
        - large_ap (bool): If True, simulate with a large aperture (used to cover larger objects)
    Returns: None
    """
    t1 = time.time()
    # Run simulation
    simulator = Simulator(rotated_mesh, radar_type, simulate_specularity=simulate_specularity, 
                          specularity_normal_threshold=specularity_normal_threshold, 
                          simulate_edges=simulate_edges, edge_threshold=edge_threshold, 
                          use_large_ap=large_ap)
    simulated_image, channels, radar_shifted_locations = simulator.simulate()


    # Save image and raw wave files
    loader.save_image(radar_type, simulated_image, simulator.x_locs, simulator.y_locs, simulator.z_locs, angles, antenna_locs=radar_shifted_locations, ext=ext)
    loader.save_raw_sim_files(radar_type, channels, radar_shifted_locations, angles, ext=ext)
    t2 = time.time()
    print(f'Took {(t2-t1)} sec to finish simulation')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that  run 24ghz robot imager")
    parser.add_argument("--name", type=str, default="None" , help="Object Name. Only one of name or ID is required. Pass 'None' to autofill")
    parser.add_argument("--id", type=str, default="None", help="Object ID #. Only one of name or ID is required. Pass 'None' to autofill")
    parser.add_argument("--angles", type=utilities.parse_tuple, default=(0, 0, 0), help="A tuple of integers for mesh angle")
    parser.add_argument("--ext", type=str, default="", help="Extension to save with")
    parser.add_argument("--skip-diffuse-sim", type=bool, default=False, help="Skip the diffuse simulation?")
    parser.add_argument("--skip-specular-sim", type=bool, default=False, help="Skip the specular simulation?")
    parser.add_argument("--skip-edge-sim", type=bool, default=False, help="Skip the edge simulation?")
    parser.add_argument("--radar_type", type=str, default="77_ghz", help="Simulate the 77_ghz or 24_ghz radar?")

    args = parser.parse_args()
    obj_name = args.name
    obj_id = args.id
    ext = args.ext
    radar_type = args.radar_type

    assert radar_type in ['77_ghz', '24_ghz'], "Please choose a valid radar type."

    # Fill in missing info
    obj_info = ObjectInformation()
    obj_id, obj_name = obj_info.fill_in_identifier_sep(obj_name, obj_id)
    large_ap = obj_info.get_object_info(ExperimentAttributes.LARGE_AP, obj_name, obj_id, exp_num='1')


    num_vert=10000
    noise = 0.0
    plot_mesh = False

    # Add offsets to angle
    x_angle, y_angle, z_angle = np.array(obj_info.get_object_offset(obj_name, obj_id)) + np.array(args.angles)

    # Set up loader and visualizer with desired parameters
    loader = GenericLoader(obj_id, obj_name, is_sim=True)
    mt = MeshTransformer()
    viz = Visualizer()

    # Loads mesh file
    path_to_mesh = loader.get_path_to_uniform_mesh(num_vert=num_vert) 
    mesh = trimesh.load_mesh(path_to_mesh)
    mesh = mt.center(mesh)

    # Rotate mesh
    rotated_mesh = mt.rotate(mesh, math.radians(x_angle), math.radians(y_angle), math.radians(z_angle))
    if plot_mesh:
        rotated_mesh.show() 
        

    # Run three different simulation versions

    # SPECULAR
    if not args.skip_specular_sim:
        full_ext = f"_specular{ext}"; simulate_specularity = True; specularity_normal_threshold = 5   ; simulate_edges=False; edge_threshold = 40  
        print(f'Running with specularity? {simulate_specularity} (threshold: {specularity_normal_threshold}). Running with edges? {simulate_edges} (threshold: {edge_threshold})')
        sim(simulate_specularity, specularity_normal_threshold, full_ext, noise,rotated_mesh, args.angles, simulate_edges, edge_threshold, large_ap)

    # EDGES
    if not args.skip_edge_sim:
        full_ext = f"_edges{ext}"; simulate_specularity = False; specularity_normal_threshold = 0 ; simulate_edges=True; edge_threshold = 20 
        print(f'Running with specularity? {simulate_specularity} (threshold: {specularity_normal_threshold}). Running with edges? {simulate_edges} (threshold: {edge_threshold})')
        sim(simulate_specularity, specularity_normal_threshold, full_ext, noise,rotated_mesh, args.angles, simulate_edges, edge_threshold, large_ap)

    # DIFFUSE
    if not args.skip_diffuse_sim:
        full_ext = f"_diffuse{ext}"; simulate_specularity = False; specularity_normal_threshold = 0  ; simulate_edges=False; edge_threshold = 40  
        print(f'Running with specularity? {simulate_specularity} (threshold: {specularity_normal_threshold}). Running with edges? {simulate_edges} (threshold: {edge_threshold})')
        sim(simulate_specularity, specularity_normal_threshold, full_ext, noise,rotated_mesh, args.angles, simulate_edges, edge_threshold, large_ap)
