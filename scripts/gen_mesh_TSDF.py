# This script generates depth images from a trained model and performs TSDF
# fusion to create a mesh. It is adapted from the implementation of PGSR
# (https://github.com/zju3dv/PGSR).

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import cv2

# Add the build directory to the Python path to find pyngp
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
import pyngp as ngp

def render_depth(parent_folder, scene_name, transform_matrix):
    # Create output directories
    depth_dir = os.path.join(parent_folder, f"{scene_name}_RFL", "renders_depth")
    color_dir = os.path.join(parent_folder, f"{scene_name}_RFL", "renders_color")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    # Load the model
    snapshot_path = os.path.join(parent_folder, f"{scene_name}_RFL", "model.ingp")
    if not os.path.exists(snapshot_path):
        print(f"Error: Model snapshot not found at {snapshot_path}")
        return

    # Initialize testbed
    testbed = ngp.Testbed()
    testbed.load_snapshot(snapshot_path)
    testbed.shall_train = False

    # Get the number of training views
    n_images = testbed.nerf.training.n_images_for_training
    print(f"Found {n_images} training views in snapshot")

    # Get image resolution from the dataset metadata
    metadata = testbed.nerf.training.dataset.metadata[0]
    width = metadata.resolution[0]
    height = metadata.resolution[1]
    print(f"Using image resolution: {width}x{height}")

    # Set rendering resolution with hidden window
    testbed.init_window(width, height, hidden=True)

    # Lists to store depth images and camera poses for TSDF fusion
    depth_images = []
    color_images = []
    camera_poses = []

    # Create Open3D camera intrinsics from the first view (assuming all cameras use the same intrinsics)
    focal_length = metadata.focal_length
    principal_point = metadata.principal_point

    # Calculate fx, fy, cx, cy
    # Extract scalar values for fx, fy - the focal length appears to be an array
    fx = float(focal_length[0]) if isinstance(focal_length, np.ndarray) else float(focal_length)
    fy = float(focal_length[1]) if isinstance(focal_length, np.ndarray) and len(focal_length) > 1 else fx
    cx = float(width * principal_point[0])
    cy = float(height * principal_point[1])

    print(f"Using camera intrinsics:")
    print(f"  Focal length: {focal_length}")
    print(f"  Principal point: {principal_point}")
    print(f"  fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    voxel_size = 0.004  # Needs to be adjusted based on the scene scale
    trunc_dist = 4 * voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=trunc_dist,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Process each frame
    for i in tqdm(range(n_images), desc="Processing frames"):
        # Set camera pose in testbed
        # INGP uses camera-to-world matrix
        testbed.set_camera_to_training_view(i)

        # Get camera extrinsics from the current view (c2w matrix)
        c2w = testbed.nerf.training.get_camera_extrinsics(i)
        c2w_4x4 = np.vstack((c2w, np.array([0, 0, 0, 1])))

        # Step 1: Invert c2w to get world-to-camera matrix
        w2c = np.linalg.inv(c2w_4x4)

        # Apply transformation for Open3D's coordinate system
        pose = transform_matrix @ w2c

        # Debug the shape of pose
        if i == 0:
            print(f"Pose matrix: \n{pose}")

        # First render color image
        testbed.render_mode = ngp.RenderMode.Shade
        color = testbed.render(width, height, spp=1)

        # Make sure color is in the right format (RGBA -> RGB)
        if color.shape[2] == 4:
            color = color[:, :, :3]

        # Gamma correct color
        color = color ** (1.0 / 2.2)

        # Convert float color to uint8
        if color.dtype != np.uint8:
            color = (color * 255).astype(np.uint8)

        # Now render depth
        testbed.render_mode = ngp.RenderMode.Depth
        depth = testbed.render(width, height, spp=1)

        # Convert depth to meters
        depth = depth.squeeze()

        # Store depth image and camera pose for TSDF fusion
        depth_images.append(depth)
        color_images.append(color)
        camera_poses.append(pose)

        # Save visualized depth image (for debugging)
        depth_vis_path = os.path.join(depth_dir, f"{i:06d}.png")
        # Ensure depth is a 2D array
        if len(depth.shape) > 2:
            depth = depth.squeeze()
        # Normalize to [0, 255] and convert to uint8
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_norm = (depth_norm * 255).clip(0, 255).astype(np.uint8)
        # Ensure it's a single-channel image
        if len(depth_norm.shape) == 3:
            depth_norm = depth_norm[:, :, 0]
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(depth_vis_path, depth_color)

        # Save color image
        color_path = os.path.join(color_dir, f"{i:06d}.png")
        cv2.imwrite(color_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR

    # Perform TSDF fusion
    print("\nPerforming TSDF fusion...")
    print(f"Using voxel size: {volume.voxel_length:.3f} m")
    print(f"Using truncation distance: {volume.sdf_trunc:.3f} m")

    # Integrate each depth image
    for i, (depth, color, pose) in enumerate(tqdm(zip(depth_images, color_images, camera_poses), desc="Integrating depth images")):
        # Process depth like PGSR
        depth = depth.copy()

        # If depth has 4 channels (RGBA), take only the first channel
        if len(depth.shape) == 3 and depth.shape[2] == 4:
            # The depth format is DDDA
            depth = depth[:, :, 0]

        # Save original depth statistics before thresholding
        valid_pixels_before = np.sum(depth > 0)
        min_depth_before = depth[depth > 0].min() if np.sum(depth > 0) > 0 else 0
        max_depth_before = depth.max()

        # Apply depth threshold - increasing from 5.0 to 10.0 meters to capture more of the scene
        depth_threshold = 10.0  # Increased from 5.0
        depth[depth > depth_threshold] = 0

        # Display depth statistics after thresholding
        valid_pixels_after = np.sum(depth > 0)
        valid_percent = 100 * valid_pixels_after / depth.size

        # Convert depth to millimeters (matching PGSR)
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_mm)
        color_o3d = o3d.geometry.Image(color)

        # Create RGBD image with adjusted parameters
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=depth_threshold,
            convert_rgb_to_intensity=False
        )

        if i % 10 == 0:
            print(f"\nIntegrating frame {i}/{len(depth_images)}")
            print(f"  Depth range: {min_depth_before:.3f} - {max_depth_before:.3f} meters")
            print(f"  Valid pixels before threshold: {valid_pixels_before} / {depth.size} ({100 * valid_pixels_before / depth.size:.1f}%)")
            print(f"  Valid pixels after threshold: {valid_pixels_after} / {depth.size} ({valid_percent:.1f}%)")

        # Integrate depth image
        volume.integrate(
            rgbd,
            intrinsic,
            pose  # Use the Open3D pose directly
        )

    # Extract mesh
    print("\nExtracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()

    # Debug: Print mesh statistics
    print("\nMesh statistics before post-processing:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of triangles: {len(mesh.triangles)}")

    # Save mesh
    mesh_path = os.path.join(parent_folder, f"{scene_name}_RFL", "tsdf_fusion.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"TSDF fusion mesh saved to {mesh_path}")

    # Post-process mesh like PGSR
    print("\nPost-processing mesh...")
    mesh = post_process_mesh(mesh, cluster_to_keep=1)
    mesh_path_post = os.path.join(parent_folder, f"{scene_name}_RFL", "tsdf_fusion_post.ply")
    o3d.io.write_triangle_mesh(mesh_path_post, mesh)
    print(f"Post-processed mesh saved to {mesh_path_post}")

    # Debug: Print final mesh statistics
    print("\nMesh statistics after post-processing:")
    print(f"  Number of vertices: {len(mesh.vertices)}")
    print(f"  Number of triangles: {len(mesh.triangles)}")

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("Post-processing the mesh to have {} clusters".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

# Mapping of scan numbers to scene names
scan_to_scene = {
    24: ["dtu_red"],
    37: ["dtu_scissors"],
    40: ["dtu_stonehenge"],
    55: ["dtu_bunny"],
    63: ["dtu_fruit"],
    65: ["dtu_skull"],
    69: ["dtu_christmas"],
    83: ["dtu_smurfs"],
    97: ["dtu_can"],
    105: ["dtu_toy"],
    106: ["dtu_pigeon"],
    110: ["dtu_gold"],
    114: ["dtu_buddha"],
    118: ["dtu_angel"],
    122: ["dtu_chouette"]
}

def main():
    parser = ArgumentParser(description="Generate depth images from a trained model")
    parser.add_argument("--parent_folder", required=True, help="Path to the trained model directory")
    parser.add_argument("--scene_name", required=True, help="Folder scene_name")

    args = parser.parse_args()

    # Use opencv_to_open3d transformation as it's known to work
    transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    render_depth(args.parent_folder, args.scene_name, transform)

if __name__ == "__main__":
    main()
