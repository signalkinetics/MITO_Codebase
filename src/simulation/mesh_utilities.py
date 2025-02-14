"""
This file contains utility functions for both transforming and visualizing meshes.
"""

import trimesh
import yaml
import math
import numpy as np
import pyrender
import matplotlib.pyplot as plt

class MeshTransformer:
    """
    MeshTransformer includes helper functions to transform and get information about the given mesh file
    """
    def translate(self, mesh, x=0.0, y=0.0, z=0.0):
        """
        Parameters:
            mesh (trimesh object): mesh to translate
            x (double): x-coordinate of the translation vector
            y (double): y-coordinate of the translation vector
            z (double): z-coordinate of the translation vector

        Returns:
            Translated copy of mesh based on the given vector
        """
        trans_matrix = trimesh.transformations.translation_matrix([x, y, z])
        mesh = mesh.copy()
        mesh.apply_transform(trans_matrix)
        return mesh

    def rotate(self, mesh, theta_x=0.0, theta_y=0.0, theta_z=0.0,
                           pivot_x=0.0, pivot_y=0.0, pivot_z=0.0):
        """
        Parameters:
            mesh (trimesh object): mesh to rotate
            theta_x, theta_y, theta_z (double): rotation along the x/y/z-axis in radians
            pivot_x, pivot_y, pivot_z (double): x/y/z-coordinate of the pivot

        Returns:
            Rotated copy of mesh based on the given angles and pivot point
        """
        rot_matrix_x = trimesh.transformations.rotation_matrix(
            theta_x, [1, 0, 0], [pivot_x, pivot_y, pivot_z]
        )
        rot_matrix_y = trimesh.transformations.rotation_matrix(
            theta_y, [0, 1, 0], [pivot_x, pivot_y, pivot_z]
        )
        rot_matrix_z = trimesh.transformations.rotation_matrix(
            theta_z, [0, 0, 1], [pivot_x, pivot_y, pivot_z]
        )
        mesh = mesh.copy()
        mesh.apply_transform(rot_matrix_x)
        mesh.apply_transform(rot_matrix_y)
        mesh.apply_transform(rot_matrix_z)
        return mesh

    def scale(self, mesh, scalar):
        """
        Parameters:
            mesh (trimesh object): mesh to scale
            scalar (float): how much to scale the mesh

        Returns:
            Mesh object that is scaled by the value of 'scalar'
        """
        matrix = np.eye(4)
        matrix[:3, :3] *= scalar
        mesh = mesh.copy()
        mesh.apply_transform(matrix)
        return mesh

    def center(self, mesh):
        """
        Parameters:
            mesh (trimesh object): mesh to center

        Returns:
            Translated copy of mesh to align the center of its bounding box to the origin
        """
        return self.translate(
            mesh, *[-(mesh.bounds[0][i] + mesh.bounds[1][i]) / 2 for i in range(3)]
        )

    def transform(self, mesh, path):
        """
        Parameters:
            mesh (trimesh object): mesh to transform
            path (str): the path to the yaml file
        
        Returns:
            Transformed mesh file based on the given transform.yaml file
        """
        with open(path, "r") as f:
            mesh_transform = yaml.safe_load(f)
        mesh = self.rotate(mesh, *mesh_transform["rotation"].values())
        mesh = self.translate(mesh, *mesh_transform["translation"].values())
        return mesh

    def find_bounding_box(self, mesh, margin=0.5):
        """
        Finds the bounding box of the given mesh and adds margin% to each dimension

        Parameters:
            mesh (trimesh object): mesh to find the bounding box
        """
        return mesh.bounds.transpose() * (1 + margin)


class MeshVisualizer:
    """
    MeshVisualizer includes helper functions to visualize meshes
    """
    def generate_2d_from_mesh(self, mesh, camera_pose=[[1, 0, 0,  0 ],
                                                       [0, 1, 0,  0 ],
                                                       [0, 0, 1, 0.5],
                                                       [0, 0, 0,  1 ]]):
        """
        Generates a 2d image of the given mesh file based on the camera pose

        Parameters:
            mesh (trimesh object): mesh to view
            camera_pose (4x4 array): 4x4 homogeneous transformation matrix that describes the pose of the camera

        Returns:
            Tuple of color buffer in RGB format (shape=(height, width, 3)) and depth buffer
        """
        render_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=math.radians(42))
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e4)
        scene.add(render_mesh, pose=np.eye(4))
        scene.add(light, pose=camera_pose)
        scene.add(camera, pose=camera_pose)

        # Calculate color and depth
        r = pyrender.OffscreenRenderer(512, 512)
        color, depth = r.render(scene)

        # Camera clipping and scaling
        min_depth = 0.1
        max_depth = depth.max() + 0.02
        depth[depth == 0] = max_depth
        np.clip(depth, min_depth, max_depth)
        depth *= 1000

        return color, depth

    def show_rgbd(self, color, depth):
        """
        Plots the rgbd data generated by the generate_2d_from_mesh function

        Parameters:
            color (numpy array): generated color buffer
            depth (numpy array): generated depth buffer
        """
        plt.figure(figsize=(8,8), frameon=False)
        plt.axis("off")
        plt.imshow(color)
        plt.show()

        plt.figure(figsize=(8,8), frameon=False)
        plt.axis("off")
        plt.imshow(depth, cmap=plt.cm.gray_r)
        plt.show()
        plt.close("all")