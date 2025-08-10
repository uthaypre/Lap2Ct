import open3d as o3d
import numpy as np
from skimage import measure
import nibabel as nib
import logging
from pathlib import Path

def get_3dct(ct_path, label_classes=None, label_colors=None):
    ct_mask = nib.load(ct_path).get_fdata()

    organ_collection = {}
    organ_bool = []
    for label, organ in label_classes.items():
        organ_mask = (ct_mask == int(label)).astype(np.uint8)  # Create a mask for the specific organ
        # Check if organ exists in the data
        if np.sum(organ_mask) == 0:
            print(f"Skipping {organ} (label {label}) - not present in this scan")
            organ_bool.append(False)
            organ_collection[organ] = None
            continue
        organ_bool.append(True)
        verts, faces, _, _ = measure.marching_cubes(organ_mask, level=0.5)  # Extract the mesh for the organ

        organ_mesh = o3d.geometry.TriangleMesh()
        organ_mesh.vertices = o3d.utility.Vector3dVector(verts)
        organ_mesh.triangles = o3d.utility.Vector3iVector(faces)
        organ_mesh.compute_vertex_normals()
        organ_mesh.paint_uniform_color(label_colors[label])  # Assign a random color to each organ
        organ_collection[organ] = organ_mesh  # Store the organ name and mesh
        print(f"Organ: {organ}, label: {label}")

    # Visualize the organ meshes
    # o3d.visualization.draw_geometries([organ_collection[organ] for organ in organ_collection])

    ## save point cloud
    # o3d.io.write_triangle_mesh(ct_path.parent / (Path(ct_path.stem).stem + ".obj"), [organ_collection[organ] for organ in organ_collection])  # Save the liver mesh as an example
    return organ_collection, organ_bool # both have length = 25