"""
This file can be used for the installation of this repository. 

Run with the --install flag to install the necessary requirements. 
If run without the --install flag, it will only compile the C++/CUDA code.

It is advised to run this script inside a virtual environment.
"""


import sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description="A script that process robot collected data to create mmWave image")
parser.add_argument("--install", action='store_true', help="If flag is provided, install necessary packages. Otherwise, only compile C++/CUDA code")
parser.add_argument("--download-data", action='store_true', help="If flag is provided, download sample data. Otherwise, only compile C++/CUDA code")
args = parser.parse_args()

# Only run install if requested 
if args.install:
    # Install main requirements
    proc = subprocess.Popen("pip install -r requirements.txt;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
    proc.wait()

    # Install git LFS
    proc = subprocess.Popen("./install_git_lfs.sh", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/utils")
    proc.wait()


    # Install additional requirements as needed
    from src.utils import utilities
    if utilities.load_param_json()['processing']['use_cuda']:
        proc = subprocess.Popen("pip install pycuda", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()
    if utilities.load_param_json()['processing']['use_simulation']:
        proc = subprocess.Popen("pip install git+https://github.com/MPI-IS/mesh.git", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()
    
    if utilities.load_param_json()['processing']['use_segmentation']:
        proc = subprocess.Popen("pip install git+https://github.com/facebookresearch/segment-anything.git", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd=".")
        proc.wait()
        
# Build the C++/CUDA Code
proc = subprocess.Popen("make;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/simulation/cpp")
proc.wait() 
proc = subprocess.Popen("make;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/data_processing/cpp")
proc.wait()
if utilities.load_param_json()['processing']['use_cuda']:
    proc = subprocess.Popen("nvcc --cubin -arch sm_86 --std=c++11 imaging_gpu.cu -o imaging_gpu.o;", shell=True, stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(), cwd="./src/data_processing/cuda")
    proc.wait()
