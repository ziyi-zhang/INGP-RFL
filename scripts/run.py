#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.	All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json
import shutil
from datetime import datetime

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa


def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

	parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--save_snapshot", action="store_true", default=False, help="Save this snapshot after training.")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	parser.add_argument("--test_transforms", action="store_true", default=False, help="Use test transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")
	parser.add_argument("--screenshot_with_normals", action="store_true", help="Render additional screenshots with normals.")
	parser.add_argument("--screenshot_depth", action="store_true", help="Render screenshots with depth.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", default=True, help="Run the testbed GUI interactively.")
	parser.add_argument("--no-gui", dest="gui", action="store_false", help="Run the testbed without GUI.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")
	parser.add_argument("--training_mode", default="RFL", choices=["RFL", "NeRF", "RFLrelax"], help="Defines the training mode.")
	parser.add_argument("--seconds", type=int, default=-1, help="Time in seconds spent on training for before quitting (only available when GUI is turned off).")
	parser.add_argument("--save_losses", action="store_true", default=False, help="Save all losses to a file after training. (saved as a numpy zipped archive .npz)")
	parser.add_argument("--save_avg_samples_per_ray", action="store_true", default=False, help="Save the avg. number of samples per ray used at each iteration to a file after training. (saved as a numpy zipped archive .npz)")

	parser.add_argument("--identifier", default="", help="Identifier that will be used to store all run outptus")
	parser.add_argument("--folder", default="", help="Folder to store all run outputs")

	parser.add_argument("--loss_type", default="Default", choices=["L1", "L2", "Default"])

	# RFL-related arguments
	reversed_train_group = parser.add_mutually_exclusive_group()
	reversed_train_group.add_argument("--reversed_train", dest="reversed_train", action="store_true", default=None)
	reversed_train_group.add_argument("--no_reversed_train", dest="reversed_train", action="store_false", default=None)
	parser.add_argument("--throughput_thres", default=None, type=float)
	random_dropout_group = parser.add_mutually_exclusive_group()
	random_dropout_group.add_argument("--random_dropout", dest="random_dropout", action="store_true", default=None)
	random_dropout_group.add_argument("--no_random_dropout", dest="random_dropout", action="store_false", default=None)
	parser.add_argument("--random_dropout_thres", default=None, type=float)
	adjust_transmittance_group = parser.add_mutually_exclusive_group()
	adjust_transmittance_group.add_argument("--adjust_transmittance", dest="adjust_transmittance", action="store_true", default=None)
	adjust_transmittance_group.add_argument("--no_adjust_transmittance", dest="adjust_transmittance", action="store_false", default=None)
	parser.add_argument("--adjust_transmittance_strength", default=None, type=float)
	parser.add_argument("--adjust_transmittance_thres", default=None, type=float)
	mw_warm_start_group = parser.add_mutually_exclusive_group()
	mw_warm_start_group.add_argument("--mw_warm_start", dest="mw_warm_start", action="store_true", default=None)
	mw_warm_start_group.add_argument("--no_mw_warm_start", dest="mw_warm_start", action="store_false", default=None)
	parser.add_argument("--mw_warm_start_steps", default=None, type=int)
	early_density_suppression_group = parser.add_mutually_exclusive_group()
	early_density_suppression_group.add_argument("--early_density_suppression", dest="early_density_suppression", action="store_true", default=None)
	early_density_suppression_group.add_argument("--no_early_density_suppression", dest="early_density_suppression", action="store_false", default=None)
	parser.add_argument("--early_density_suppression_end", default=None, type=int)
	parser.add_argument("--laplacian_mode", default="Disabled", choices=["Disabled", "Volume", "Surface"])
	laplacian_weight_decay_group = parser.add_mutually_exclusive_group()
	laplacian_weight_decay_group.add_argument("--laplacian_weight_decay", dest="laplacian_weight_decay", action="store_true", default=None)
	laplacian_weight_decay_group.add_argument("--no_laplacian_weight_decay", dest="laplacian_weight_decay", action="store_false", default=None)
	parser.add_argument("--laplacian_weight_decay_steps", default=None, type=int)
	parser.add_argument("--laplacian_weight_decay_strength", default=None, type=float)
	parser.add_argument("--laplacian_weight_decay_min", default=None, type=float)
	laplacian_candidate_on_grid_group = parser.add_mutually_exclusive_group()
	laplacian_candidate_on_grid_group.add_argument("--laplacian_candidate_on_grid", dest="laplacian_candidate_on_grid", action="store_true", default=None)
	laplacian_candidate_on_grid_group.add_argument("--no_laplacian_candidate_on_grid", dest="laplacian_candidate_on_grid", action="store_false", default=None)
	parser.add_argument("--laplacian_fd_epsilon", default=None, type=float)
	parser.add_argument("--laplacian_weight", default=None, type=float)
	parser.add_argument("--laplacian_density_thres", default=None, type=float)
	parser.add_argument("--refinement_start", default=None, type=int)
	parser.add_argument("--laplacian_refinement_strength", default=None, type=float)
	parser.add_argument("--surface_threshold", default=None, type=float)
	parser.add_argument("--crop_size", default=None, type=float)

	reflected_dir_group = parser.add_mutually_exclusive_group()
	reflected_dir_group.add_argument("--reflected_dir", dest="reflected_dir", action="store_true", default=None)
	reflected_dir_group.add_argument("--no_reflected_dir", dest="reflected_dir", action="store_false", default=None)


	return parser.parse_args()

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

if __name__ == "__main__":
	args = parse_args()
	if args.vr: # VR implies having the GUI running at the moment
		args.gui = True

	identifier = None
	if args.identifier != "":
		# e.g., replace "scene-training_mode" with f"{args.scene}_{args.training_mode}"
		parts = args.identifier.split("-")
		formatted_parts = [getattr(args, part) if hasattr(args, part) else part
								for part in parts]
		identifier = "_".join(formatted_parts)

		if args.folder:
			output_folder = os.path.join("out", args.folder, identifier)
		else:
			identifier = identifier + f"_{datetime.now().isoformat(timespec='seconds')}"
			output_folder = f"./out/{identifier}"
		print(f"Writing all outputs to \"{output_folder}\"")
		os.makedirs(os.path.dirname(output_folder + "/base.json"), exist_ok=True)
		shutil.copyfile("configs/nerf/base.json", output_folder + "/base.json")
	else:
		output_folder = args.folder

	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	for file in args.files:
		scene_info = get_scene(file)
		if scene_info:
			file = os.path.join(scene_info["data_dir"], scene_info["dataset_train"])
		testbed.load_file(file)

	if args.scene:
		original_scene_name = args.scene
		scene_info = get_scene(args.scene)
		if scene_info is not None:
			args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset_train"])
			if not args.network and "network" in scene_info:
				args.network = scene_info["network"]

		testbed.load_training_data(args.scene)

		if args.identifier:
			with open(output_folder + '/original_scene_name.txt', 'w') as original_scene_name_f:
				original_scene_name_f.write(original_scene_name)


	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window)
		if args.vr:
			testbed.init_vr()


	if args.load_snapshot:
		scene_info = get_scene(args.load_snapshot)
		if scene_info is not None:
			args.load_snapshot = default_snapshot_filename(scene_info)
		testbed.load_snapshot(args.load_snapshot)
	elif args.network:
		testbed.reload_network_from_file(args.network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_lens_distortion = True

	network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
	if testbed.mode == ngp.TestbedMode.Sdf:
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Match nerf paper behaviour and train on a fixed bg.
		testbed.nerf.training.random_bg_color = False

	if args.loss_type == "L1":
		testbed.nerf.training.loss_type = ngp.LossType.L1
		testbed.nerf.training.ignore_json_loss = True
		print(f"Using loss type {testbed.nerf.training.loss_type}")
	elif args.loss_type == "L2":
		testbed.nerf.training.loss_type = ngp.LossType.L2
		testbed.nerf.training.ignore_json_loss = True
		print(f"Using loss type {testbed.nerf.training.loss_type}")
	else:
		pass  # Use json loss

	if args.training_mode == 'RFL':
		testbed.set_train_mode(ngp.TrainMode.RFL);
		testbed.set_surface_rendering(True);
	elif args.training_mode == 'RFLrelax':
		testbed.set_train_mode(ngp.TrainMode.RFLrelax);
		testbed.set_surface_rendering(False);
	elif args.training_mode == 'NeRF':
		testbed.set_train_mode(ngp.TrainMode.NeRF);
		testbed.set_surface_rendering(False);
	else:
		raise RuntimeError(f"Unsupported training mode: {args.training_mode}")

	if args.reversed_train is not None:
		testbed.nerf.training.reversed_train = args.reversed_train
	if args.throughput_thres is not None:
		testbed.nerf.training.throughput_thres = args.throughput_thres
	if args.random_dropout is not None:
		testbed.nerf.training.random_dropout = args.random_dropout
	if args.random_dropout_thres is not None:
		testbed.nerf.training.random_dropout_thres = args.random_dropout_thres
	if args.adjust_transmittance is not None:
		testbed.nerf.training.adjust_transmittance = args.adjust_transmittance
	if args.adjust_transmittance_strength is not None:
		testbed.nerf.training.adjust_transmittance_strength = args.adjust_transmittance_strength
	if args.adjust_transmittance_thres is not None:
		testbed.nerf.training.adjust_transmittance_thres = args.adjust_transmittance_thres
	if args.mw_warm_start is not None:
		testbed.nerf.training.mw_warm_start = args.mw_warm_start
	if args.mw_warm_start_steps is not None:
		testbed.nerf.training.mw_warm_start_steps = args.mw_warm_start_steps
	if args.early_density_suppression is not None:
		testbed.nerf.training.early_density_suppression = args.early_density_suppression
	if args.early_density_suppression_end is not None:
		testbed.nerf.training.early_density_suppression_end = args.early_density_suppression_end
	if args.laplacian_mode == "Disabled":
		testbed.set_laplacian_mode(ngp.LaplacianMode.Disabled)
	elif args.laplacian_mode == "Volume":
		testbed.set_laplacian_mode(ngp.LaplacianMode.Volume)
	elif args.laplacian_mode == "Surface":
		testbed.set_laplacian_mode(ngp.LaplacianMode.Surface)
	if args.laplacian_weight_decay is not None:
		testbed.nerf.training.laplacian_weight_decay = args.laplacian_weight_decay
	if args.laplacian_weight_decay_steps is not None:
		testbed.nerf.training.laplacian_weight_decay_steps = args.laplacian_weight_decay_steps
	if args.laplacian_weight_decay_strength is not None:
		testbed.nerf.training.laplacian_weight_decay_strength = args.laplacian_weight_decay_strength
	if args.laplacian_weight_decay_min is not None:
		testbed.nerf.training.laplacian_weight_decay_min = args.laplacian_weight_decay_min
	if args.laplacian_candidate_on_grid is not None:
		testbed.nerf.training.laplacian_candidate_on_grid = args.laplacian_candidate_on_grid
	if args.laplacian_fd_epsilon is not None:
		testbed.nerf.training.laplacian_fd_epsilon = args.laplacian_fd_epsilon
	if args.laplacian_weight is not None:
		testbed.nerf.training.laplacian_weight = args.laplacian_weight
	if args.laplacian_density_thres is not None:
		testbed.nerf.training.laplacian_density_thres = args.laplacian_density_thres
	if args.refinement_start is not None:
		testbed.nerf.training.refinement_start = args.refinement_start
	if args.laplacian_refinement_strength is not None:
		testbed.nerf.training.laplacian_refinement_strength = args.laplacian_refinement_strength
	if args.surface_threshold is not None:
		testbed.nerf.surface_threshold = args.surface_threshold
	if args.reflected_dir is not None:
		testbed.nerf.reflected_dir = args.reflected_dir
	if args.crop_size is not None:
		center = (testbed.render_aabb.max + testbed.render_aabb.min) * 0.5
		scale = args.crop_size / np.max(testbed.render_aabb.max - testbed.render_aabb.min)
		testbed.render_aabb.max = np.minimum((testbed.render_aabb.max - center) * scale + center, testbed.aabb.max)
		testbed.render_aabb.min = np.maximum((testbed.render_aabb.min - center) * scale + center, testbed.aabb.min)

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and args.seconds < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	time_limited_training = False
	if n_steps < 0 and args.seconds > 0 and not args.gui:
		time_limited_training = True
		n_steps = args.seconds * 10

	if args.save_losses:
		if identifier is None:
			raise RuntimeError(f"An ID must be specified if loss values are to "
			"be saved! Specify --identifier on the command line.")
		os.makedirs(os.path.dirname(output_folder + "/loss.npz"), exist_ok=True)

	if args.save_avg_samples_per_ray:
		if identifier is None:
			raise RuntimeError(f"An ID must be specified if avg samples per ray "
			"values are to be saved! Specify --identifier on the command line.")
		os.makedirs(os.path.dirname(output_folder + "/avg_samples_per_ray.npz"), exist_ok=True)

	tqdm_last_update = 0
	losses = []
	avg_samples_per_ray = []
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="steps") as t:
			training_start_time = time.monotonic()
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)


				# What will happen when training is done?
				if not time_limited_training and testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break
				elif time_limited_training and args.seconds < (time.monotonic() - training_start_time):
					break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if args.save_losses:
					losses.append((testbed.loss, int(testbed.training_step), now - training_start_time))

				if args.save_avg_samples_per_ray:
					avg_samples_per_ray.append(
						(testbed.avg_samples_per_ray, int(testbed.training_step), now - training_start_time))

				if now - tqdm_last_update > 0.1:
					if time_limited_training:
						t.update(int((now - training_start_time) * 10) - t.n)
					else:
						t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_losses:
		loss_info = np.array(losses).transpose()
		np.savez(output_folder + "/losses.npz", losses=loss_info[0], iters=loss_info[1], time=loss_info[2])

	if args.save_avg_samples_per_ray:
		avg_smp_ray_info = np.array(avg_samples_per_ray).transpose()
		np.savez(output_folder + "/avg_samples_per_ray.npz", avg_spr=avg_smp_ray_info[0], iters=avg_smp_ray_info[1], time=avg_smp_ray_info[2])

	if args.save_snapshot:
		if identifier is None:
			raise RuntimeError(f"An ID must be specified if the model is to be "
			"saved! Specify --identifier on the command line.")
		os.makedirs(os.path.dirname(output_folder + "/model.ingp"), exist_ok=True)
		testbed.save_snapshot(output_folder + "/model.ingp", False)

	if args.test_transforms:
		if identifier is None:
			raise RuntimeError(f"An ID must be specified if a test dataset is "
			"to be evaluated! Specify --identifier on the command line.")

		rendering_test_modes = [False]
		if args.training_mode == 'RFL':
			rendering_test_modes.append(True)

		for mode_idx in range(len(rendering_test_modes)):
			testbed.set_surface_rendering(rendering_test_modes[mode_idx]);

			totmse = 0
			totpsnr = 0
			totssim = 0
			totcount = 0
			minpsnr = 1000
			maxpsnr = 0

			# Evaluate metrics on black background
			testbed.background_color = [0.0, 0.0, 0.0, 1.0]

			# Prior nerf papers don't typically do multi-sample anti aliasing.
			# So snap all pixels to the pixel centers.
			testbed.snap_to_pixel_centers = True
			spp = 8

			testbed.nerf.render_min_transmittance = 1e-4

			testbed.shall_train = False
			scene_info = get_scene(original_scene_name)
			test_transforms_path = os.path.join(scene_info["data_dir"], scene_info["dataset_test"])
			testbed.load_training_data(test_transforms_path)

			with tqdm(range(testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
				for i in t:
					resolution = testbed.nerf.training.dataset.metadata[i].resolution
					testbed.render_ground_truth = True
					testbed.set_camera_to_training_view(i)
					ref_image = testbed.render(resolution[0], resolution[1], 1, True)
					testbed.render_ground_truth = False
					image = testbed.render(resolution[0], resolution[1], spp, True)

	#				if i == 0:
	#					write_image(f"ref.png", ref_image)
	#					write_image(f"out.png", image)
	#
	#					diffimg = np.absolute(image - ref_image)
	#					diffimg[...,3:4] = 1.0
	#					write_image("diff.png", diffimg)

					if testbed.nerf.surface_rendering:
						mode_str = f"surface_{testbed.nerf.surface_threshold:.2f}"
					else:
						mode_str = "volume"
					write_image(f"{output_folder}/test_{i:04d}_{mode_str}.png", image)  # Save image as PNG

					A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
					R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
					mse = float(compute_error("MSE", A, R))
					ssim = float(compute_error("SSIM", A, R))
					totssim += ssim
					totmse += mse
					psnr = mse2psnr(mse)
					totpsnr += psnr
					minpsnr = psnr if psnr<minpsnr else minpsnr
					maxpsnr = psnr if psnr>maxpsnr else maxpsnr
					totcount = totcount+1
					t.set_postfix(psnr = totpsnr/(totcount or 1))

			psnr_avgmse = mse2psnr(totmse/(totcount or 1))
			psnr = totpsnr/(totcount or 1)
			ssim = totssim/(totcount or 1)
			print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
			np.savez(output_folder + f"/psnr_ssim_surface_rendering_{rendering_test_modes[mode_idx]}.npz", psnr=psnr, minpsnr=minpsnr, ssim=ssim)

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		thresh = args.marching_cubes_density_thresh or 2.5
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)

	if ref_transforms:
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		testbed.background_color = [1.0, 1.0, 1.0, 1.0]  # Want white background for screenshots
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f.get("transform_matrix")# f["transform_matrix_start"])
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(output_folder, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)

			# Optionally render additional images with normals
			if args.screenshot_with_normals:
				testbed.render_mode = ngp.RenderMode.Normals
				outname = os.path.splitext(outname)[0] + "_normals" + os.path.splitext(outname)[1]
				print(f"rendering {outname}")
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				write_image(outname, image)
				testbed.render_mode = ngp.RenderMode.Shade

			if args.screenshot_depth:
				testbed.render_mode = ngp.RenderMode.Depth
				outname = os.path.splitext(outname)[0] + "_depth" + os.path.splitext(outname)[1]
				print(f"rendering {outname}")
				image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
				write_image(outname, image)
				testbed.render_mode = ngp.RenderMode.Shade

	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)

		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps
		save_frames = "%" in args.video_output
		start_frame, end_frame = args.video_render_range

		if "tmp" in os.listdir():
			shutil.rmtree("tmp")
		os.makedirs("tmp")

		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			testbed.camera_smoothing = args.video_camera_smoothing

			if start_frame >= 0 and i < start_frame:
				# For camera smoothing and motion blur to work, we cannot just start rendering
				# from middle of the sequence. Instead we render a very small image and discard it
				# for these initial frames.
				# TODO Replace this with a no-op render method once it's available
				frame = testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
				continue
			elif end_frame >= 0 and i > end_frame:
				continue

			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
			if save_frames:
				write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
			else:
				write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

		if not save_frames:
			os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")

		shutil.rmtree("tmp")
