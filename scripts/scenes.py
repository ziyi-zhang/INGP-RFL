#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from common import *

def ours_real_converted(path, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, path),
		"dataset_train" : "transforms.json",
		"dataset_test"  : "transforms.json",
		"dataset"       : "",
		"test_every"    : 5,
		"frameidx"      : frameidx
	}

def nerf_synthetic(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"nerf_synthetic/{name}"),
		"dataset_train" : "transforms_train.json",
		"dataset_test"  : "transforms_test.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

def nerf_real_360(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"nerf_real_360/{name}"),
		"dataset_train" : "transforms.json",
		"dataset_test"  : "transforms.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

def mipnerf_360(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"mipnerf_360/{name}"),
		"dataset_train" : "train.json",
		"dataset_test"  : "test.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

def DTU(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"dtu/{name}"),
		"dataset_train" : "transform.json",
		"dataset_test"  : "transform.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

def bmvs(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"bmvs/{name}"),
		"dataset_train" : "transform.json",
		"dataset_test"  : "transform.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

def tnt(name, frameidx):
	return {
		"data_dir"      : os.path.join(NERF_DATA_FOLDER, f"TanksandTemples/{name}"),
		"dataset_train" : "transforms.json",
		"dataset_test"  : "transforms.json",
		"dataset"       : "",
		"frameidx"      : frameidx
	}

scenes_nerf = {
	"fox"         : ours_real_converted("fox/", frameidx=0),
	"lego"      : nerf_synthetic("lego", frameidx=52),
	"drums"     : nerf_synthetic("drums", frameidx=52),
	"ship"      : nerf_synthetic("ship", frameidx=52),
	"mic"       : nerf_synthetic("mic", frameidx=52),
	"ficus"     : nerf_synthetic("ficus", frameidx=52),
	"chair"     : nerf_synthetic("chair", frameidx=52),
	"hotdog"    : nerf_synthetic("hotdog", frameidx=52),
	"materials" : nerf_synthetic("materials", frameidx=52),

	# nerf real 360
	"pinecone" : nerf_real_360("pinecone", frameidx=0),
	"vasedeck" : nerf_real_360("vasedeck", frameidx=0),

	# mipnerf 360
	"bicycle" : mipnerf_360("bicycle", frameidx=0),
	"bonsai"  : mipnerf_360("bonsai", frameidx=0),
	"counter" : mipnerf_360("counter", frameidx=0),
	"garden"  : mipnerf_360("garden", frameidx=0),
	"kitchen" : mipnerf_360("kitchen", frameidx=0),
	"room"    : mipnerf_360("room", frameidx=0),
	"stump"   : mipnerf_360("stump", frameidx=0),
	"treehill": mipnerf_360("treehill", frameidx=0),
	"flowers" : mipnerf_360("flowers", frameidx=0),

	# DTU
    "dtu24": DTU("dtu_scan24", frameidx=0),
    "dtu_maison": DTU("dtu_scan24", frameidx=0),  # Alias name
	"dtu_red": DTU("dtu_scan24", frameidx=0),
    "dtu37": DTU("dtu_scan37", frameidx=0),
    "dtu_scissors": DTU("dtu_scan37", frameidx=0),
    "dtu40": DTU("dtu_scan40", frameidx=0),
    "dtu_stonehenge": DTU("dtu_scan40", frameidx=0),
    "dtu55": DTU("dtu_scan55", frameidx=0),
    "dtu_bunny": DTU("dtu_scan55", frameidx=0),
    "dtu63": DTU("dtu_scan63", frameidx=0),
    "dtu_fruit": DTU("dtu_scan63", frameidx=0),
    "dtu65": DTU("dtu_scan65", frameidx=0),
    "dtu_skull": DTU("dtu_scan65", frameidx=0),
    "dtu69": DTU("dtu_scan69", frameidx=0),
    "dtu_christmas": DTU("dtu_scan69", frameidx=0),
    "dtu83": DTU("dtu_scan83", frameidx=0),
    "dtu_smurfs": DTU("dtu_scan83", frameidx=0),
    "dtu97": DTU("dtu_scan97", frameidx=0),
    "dtu_can": DTU("dtu_scan97", frameidx=0),
    "dtu105": DTU("dtu_scan105", frameidx=0),
    "dtu_toy": DTU("dtu_scan105", frameidx=0),
    "dtu106": DTU("dtu_scan106", frameidx=0),
    "dtu_pigeon": DTU("dtu_scan106", frameidx=0),
    "dtu110": DTU("dtu_scan110", frameidx=0),
    "dtu_gold": DTU("dtu_scan110", frameidx=0),
    "dtu114": DTU("dtu_scan114", frameidx=0),
    "dtu_buddha": DTU("dtu_scan114", frameidx=0),
    "dtu118": DTU("dtu_scan118", frameidx=0),
    "dtu_angel": DTU("dtu_scan118", frameidx=0),
    "dtu122": DTU("dtu_scan122", frameidx=0),
    "dtu_chouette": DTU("dtu_scan122", frameidx=0),

	# BMVS
    "bm_bear": bmvs("bmvs_bear", frameidx=0),
    "bm_clock": bmvs("bmvs_clock", frameidx=0),
	"bm_dog": bmvs("bmvs_dog", frameidx=0),
    "bm_durian": bmvs("bmvs_durian", frameidx=0),
    "bm_jade": bmvs("bmvs_jade", frameidx=0),
    "bm_man": bmvs("bmvs_man", frameidx=0),
    "bm_sculpture": bmvs("bmvs_sculpture", frameidx=0),
    "bm_stone": bmvs("bmvs_stone", frameidx=0),

	# Tanks and Temples
	"tt_barn": tnt("Barn", frameidx=0),
	"tt_truck": tnt("Truck", frameidx=0),
	"tt_caterpillar": tnt("Caterpillar", frameidx=0),
	"tt_biglego": tnt("Caterpillar", frameidx=0),  # Alias name
	"tt_church": tnt("Church", frameidx=0),
	"tt_courthouse": tnt("Courthouse", frameidx=0),
	"tt_ignatius": tnt("Ignatius", frameidx=0),
	"tt_meetingroom": tnt("Meetingroom", frameidx=0),
	"tt_cafe": tnt("Meetingroom", frameidx=0),  # Alias name
}

def ours_mesh(name, up = [0,1,0], infolder=True):
	return {
		"data_dir"      : os.path.join(SDF_DATA_FOLDER, f"{name}") if infolder else SDF_DATA_FOLDER,
		"dataset"       : f"{name}.obj",
		"up"            : up
	}

scenes_sdf = {
	"armadillo"     : ours_mesh("armadillo", infolder=False),
}

def ours_image(name, infolder=True):
	data_dir = os.path.join(IMAGE_DATA_FOLDER, f"{name}") if infolder else IMAGE_DATA_FOLDER
	dataset = f"{name}.bin"
	if not os.path.exists(os.path.join(data_dir, dataset)):
		dataset = f"{name}.exr"
		if not os.path.exists(os.path.join(data_dir, dataset)):
			dataset = f"{name}.png"
			if not os.path.exists(os.path.join(data_dir, dataset)):
				dataset = f"{name}.jpg"

	return {
		"data_dir"      : data_dir,
		"dataset"       : dataset
	}

scenes_image = {
	"albert"        : ours_image("albert", False),
}


def ours_volume(name, ds):
	return {
		"data_dir" : os.path.join(VOLUME_DATA_FOLDER, f"{name}"),
		"dataset" : ds
	}

scenes_volume = {
}

def setup_colored_sdf(testbed, scene, softshadow=True):
	if scene == "lizard":
		testbed.background_color = [0.882, 0.580, 0.580, 1.000]
		testbed.exposure = 1.000
		testbed.sun_dir=[-0.325,0.590,0.738]
		testbed.up_dir=[0.000,1.000,0.000]
		testbed.view_dir=[-0.366,-0.314,-0.876]
		testbed.look_at=[0.587,0.420,0.479]
		testbed.scale=0.887
		testbed.fov,testbed.aperture_size,testbed.slice_plane_z=39.600,0.000,0.655

		testbed.sdf.brdf.metallic=0.000
		testbed.sdf.brdf.subsurface=0.000
		testbed.sdf.brdf.specular=1.000
		testbed.sdf.brdf.roughness=0.300
		testbed.sdf.brdf.sheen=0.000
		testbed.sdf.brdf.clearcoat=0.000
		testbed.sdf.brdf.clearcoat_gloss=0.000
		testbed.sdf.brdf.basecolor=[0.800,0.800,0.800]

	elif scene == "cow":
		testbed.background_color = [0.580, 0.882, 0.607, 1.000]
		testbed.exposure = 0.5
		testbed.sun_dir=[-0.604,0.491,0.386]
		testbed.up_dir=[0.000,1.000,0.000]
		testbed.view_dir=[0.997,-0.059,-0.057]
		testbed.look_at=[0.500,0.500,0.500]
		testbed.fov,testbed.aperture_size,testbed.slice_plane_z=40.700,0.000,0.557
		testbed.scale=0.976

		testbed.sdf.brdf.metallic=0.000
		testbed.sdf.brdf.subsurface=0.000
		testbed.sdf.brdf.specular=1.000
		testbed.sdf.brdf.roughness=0.300
		testbed.sdf.brdf.sheen=0.000
		testbed.sdf.brdf.clearcoat=0.000
		testbed.sdf.brdf.clearcoat_gloss=0.000
		testbed.sdf.brdf.basecolor=[0.800,0.800,0.800]

	elif scene == "clockwork":
		testbed.background_color = [0.882, 0.731, 0.580, 1.000]
		testbed.exposure = 3.000
		testbed.sun_dir=[-0.236,0.946,-0.220]
		testbed.up_dir=[0.000,1.000,0.000]
		testbed.view_dir=[-0.639,-0.720,0.272]
		testbed.look_at=[0.540,0.451,0.457]
		testbed.scale=1.074
		testbed.fov,testbed.aperture_size,testbed.slice_plane_z=39.600,0.000,0.025

		testbed.sdf.brdf.metallic=1.000
		testbed.sdf.brdf.subsurface=0.000
		testbed.sdf.brdf.specular=1.000
		testbed.sdf.brdf.roughness=0.300
		testbed.sdf.brdf.sheen=0.000
		testbed.sdf.brdf.clearcoat=0.000
		testbed.sdf.brdf.clearcoat_gloss=0.000
		testbed.sdf.brdf.basecolor=[0.800,0.800,0.800]

	elif scene == "lucy":
		testbed.background_color = [0.597, 0.797, 0.697, 1.000]
		testbed.exposure = 1.000
		testbed.sun_dir=[0.290,0.342,0.893]
		testbed.up_dir=[0.000,0.000,1.000]
		testbed.view_dir=[0.003,-0.960,-0.281]
		testbed.scale=1.299
		testbed.fov,testbed.aperture_size,testbed.slice_plane_z=39.600,0.000,0.768

		testbed.sdf.brdf.metallic=0.000
		testbed.sdf.brdf.subsurface=0.000
		testbed.sdf.brdf.specular=0.194
		testbed.sdf.brdf.roughness=0.300
		testbed.sdf.brdf.sheen=0.000
		testbed.sdf.brdf.clearcoat=0.000
		testbed.sdf.brdf.clearcoat_gloss=0.000
		testbed.sdf.brdf.basecolor=[0.800,0.800,0.800]
		softshadow = True

	else:
		if scene == "bearded_man":
			testbed.up_dir=[0.000,-1.000,-0.000]
			testbed.view_dir=[-0.924,0.128,-0.361]
			testbed.look_at=[0.500,0.500,0.500]
			testbed.fov,testbed.aperture_size,testbed.slice_plane_z=39.600,0.000,0.377

		testbed.background_color = [0.580, 0.713, 0.882, 1.000]
		testbed.exposure = 1.000
		testbed.sun_dir=[0.541,-0.839,-0.042]

		testbed.sdf.brdf.metallic=0.000
		testbed.sdf.brdf.subsurface=0.000
		testbed.sdf.brdf.specular=1.000
		testbed.sdf.brdf.roughness=0.300
		testbed.sdf.brdf.sheen=0.000
		testbed.sdf.brdf.clearcoat=0.000
		testbed.sdf.brdf.clearcoat_gloss=0.000
		testbed.sdf.brdf.basecolor=[0.800,0.800,0.800]

	testbed.autofocus_target=[0.500,0.500,0.500]
	testbed.autofocus = False

	testbed.sdf.analytic_normals = False
	testbed.sdf.use_triangle_octree = False

	col = list(testbed.background_color)
	testbed.sdf.brdf.ambientcolor = np.multiply(col,col)[0:3]
	testbed.sdf.shadow_sharpness = 16 if softshadow else 2048
	testbed.scale = testbed.scale * 1.13

def default_snapshot_filename(scene_info):
	filename = f"base.ingp"
	if scene_info["dataset"]:
		filename = f"{os.path.splitext(scene_info['dataset'])[0]}_{filename}"
	return os.path.join(scene_info["data_dir"], filename)
