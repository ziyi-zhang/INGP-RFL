#!/bin/bash

# This script runs the RFL algorithm with a simple Laplacian regularization for
# geoemtry optimization.

# List of scene names
declare -A datasets
datasets["synthetic"]="synthetic_scenes"
datasets["dtu"]="dtu_scenes"
datasets["bm"]="bm_scenes"
datasets["mipnerf"]="mipnerf_scenes"
datasets["custom"]="custom_scenes"

synthetic_scenes=("lego" "drums" "ship" "mic" "ficus" "chair" "hotdog" "materials")
dtu_scenes=("dtu_red" "dtu_scissors" "dtu_stonehenge" "dtu_bunny" "dtu_fruit" "dtu_skull" "dtu_christmas" "dtu_smurfs" "dtu_can" "dtu_toy" "dtu_pigeon" "dtu_gold" "dtu_buddha" "dtu_angel" "dtu_chouette")
bm_scenes=("bm_bear" "bm_clock" "bm_dog" "bm_durian" "bm_jade" "bm_man" "bm_sculpture" "bm_stone")
mipnerf_scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
custom_scenes=("dtu_stonehenge" "bm_bear")

# Check if the user provided all necessary arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 [train|mesh|screenshot|eval] [synthetic|dtu|bm|mipnerf] [folder]"
    exit 1
fi

# GUI debug
# python3 scripts/run.py \
#     --scene bm_bear \
#     --n_steps 2000 \
#     --training_mode=RFL \
#     --mw_warm_start_steps 1000 \
#     --loss_type L1 \
#     --laplacian_mode=Volume \
#     --laplacian_weight_decay_strength 0.005 \
#     --laplacian_weight_decay_min 0.005 \
#     --refinement_start 1800 \
#     --laplacian_refinement_strength 0.00002

# # laplacian_fd_epsilon = 1.f * laplacian_fd_epsilon_min
# python3 scripts/run.py \
#     --scene dtu_pigeon \
#     --n_steps 2000 \
#     --training_mode=RFL \
#     --mw_warm_start_steps 1000 \
#     --loss_type L1 \
#     --laplacian_mode=Volume \
#     --laplacian_weight_decay_strength 0.01 \
#     --laplacian_weight_decay_min 0.005 \
#     --refinement_start 1500 \
#     --laplacian_refinement_strength 0.00002

# ./scripts/run.py \
#     --scene bonsai \
#     --n_steps 20000 \
#     --training_mode=RFL \
#     --laplacian_mode=Volume \
#     --early_density_suppression \
#     --mw_warm_start_steps 1000 \
#     --laplacian_weight_decay_strength 0.001 \
#     --laplacian_weight_decay_min 0.00002 \
#     --refinement_start 10000 \
#     --laplacian_refinement_strength 0.00002

###### BEGIN HYPERPARAMETERS
mc_res=1024
laplacian_refinement_strength=${laplacian_refinement_strength:=0.00002}
laplacian_weight_decay_min=0.005
if [ "$2" == "mipnerf" ]; then
    loss_type="Default"
    steps=20000
    refinement_start=10000
    mw_warm_start_steps=5000
    mc_threshold=4.5
    width=1920
    height=1080
elif [ "$2" == "dtu" ]; then
    loss_type="L1"
    steps=10000
    refinement_start=6000
    mw_warm_start_steps=2000
    mc_threshold=7.258  # At 2048 step size, 7.258 gives occupancy threshold of 0.5
    width=1600
    height=1200
elif [ "$2" == "bm" ]; then
    loss_type="L1"
    steps=10000
    refinement_start=6000
    mw_warm_start_steps=2000
    mc_threshold=7.258
    width=1600
    height=1200
elif [ "$2" == "synthetic" ]; then
    loss_type="Default"
    steps=10000
    refinement_start=6000
    mw_warm_start_steps=3000
    mc_threshold=7.258
    width=1600
    height=1200
else
    # custom (debug)
    loss_type="L1"
    steps=10000
    refinement_start=6000
    mw_warm_start_steps=2000
    mc_threshold=7.258
    width=1600
    height=1200
fi
####### END HYPERPARAMETERS

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
# If in train mode, append datasetname and timestamp to the given folder name
if [[ "$1" == *train* ]]; then
    folder="$3_$2_$timestamp"
else
    folder="$3"
fi

scenes_var_name="${datasets[$2]}"
eval "scenes=(\"\${${scenes_var_name}[@]}\")"
echo "Scenes: ${scenes[@]}"

if [[ "$1" == *train* ]]; then
    # Iterate over each scene
    for scene in "${scenes[@]}"; do
        echo "Processing scene: $scene"

        echo "laplacian_weight_decay_min: $laplacian_weight_decay_min"

        ./scripts/run.py \
            --scene "$scene" \
            --no-gui \
            --loss_type "$loss_type" \
            --n_steps "$steps" \
            --save_snapshot \
            --save_losses \
            --training_mode=RFL \
            --laplacian_mode=Volume \
            --early_density_suppression \
            --mw_warm_start_steps "$mw_warm_start_steps" \
            --laplacian_weight_decay_min "$laplacian_weight_decay_min" \
            --refinement_start "$refinement_start" \
            --laplacian_refinement_strength "$laplacian_refinement_strength" \
            --identifier scene-training_mode \
            --folder "$folder" \
            "${@:4}"  # Optional arguments

        # Check if the command crashed (non-zero exit status)
        if [ $? -ne 0 ]; then
            echo "Error occurred while running scene: $scene. Continuing to next scene..."
        fi

        echo "Finished scene: $scene"
    done
fi

if [[ "$1" == *mesh* ]]; then
    # Iterate over all subfolders under out/$folder and generate meshes
    for scene_dir in out/"$folder"/*; do
        if [ -d "$scene_dir" ]; then
            scene_name=$(basename "$scene_dir")
            snapshot_path="$scene_dir/model.ingp"
            mesh_output_path="$scene_dir/mesh_thres$mc_threshold.obj"

            if [ -f "$snapshot_path" ]; then
                echo "Generating mesh for $scene_name from snapshot $snapshot_path"

                ./scripts/run.py --snapshot "$snapshot_path" \
                                --save_mesh "$mesh_output_path" \
                                --marching_cubes_res $mc_res \
                                --marching_cubes_density_thresh $mc_threshold \
                                --no-gui

                if [ $? -ne 0 ]; then
                    echo "Error occurred during mesh generation for $scene_name."
                else
                    echo "Mesh generated for $scene_name."
                fi
            else
                echo "No snapshot found for $scene_name. Skipping..."
            fi
        fi
    done
fi

if [[ "$1" == *screen* ]]; then
    # Iterate over all subfolders under out/$folder and generate screenshots
    for scene_dir in out/"$folder"/*; do
        if [ -d "$scene_dir" ]; then
            scene_name=$(basename "$scene_dir")
            # Get the string before the last underscore: this should be the scene name
            scene_name=$(echo "$scene_name" | rev | cut -d'_' -f2- | rev)
            snapshot_path="$scene_dir/model.ingp"
            screenshot_output_path="$scene_dir"

            if [ -f "$snapshot_path" ]; then
                echo "Generating screenshot for $scene_name"

                ./scripts/run.py --snapshot "$snapshot_path" \
                                --screenshot_transforms misc/screenshot_trafo/"$scene_name"/trafo_static.json \
                                --screenshot_dir "$screenshot_output_path" \
                                --folder "$scene_dir" \
                                --width "$width" \
                                --height "$height" \
                                --no-gui \
                                --screenshot_with_normals \
                                --training_mode=RFL
                # "--training_mode=RFL" to enable surface rendering

                if [ $? -ne 0 ]; then
                    echo "Error occurred during screenshot generation for $scene_name."
                else
                    echo "Screenshot generated for $scene_name."
                fi
            else
                echo "No snapshot found for $scene_name. Skipping..."
            fi
        fi
    done
fi

if [ "$1" == "eval" ]; then
    # Iterate over all subfolders under out/$folder and run test transforms
    echo "Evaluating PSNR $folder !"
    for scene_dir in out/"$folder"/*; do
        if [ -d "$scene_dir" ]; then
            scene_name=$(basename "$scene_dir")
            original_scene_name=$(cat "$scene_dir/original_scene_name.txt")
            snapshot_path="$scene_dir/model.ingp"
            mesh_output_path="$scene_dir/mesh_thres$mc_threshold.obj"

            if [ -f "$snapshot_path" ]; then
                echo "Evaluating PSNR for $scene_name from snapshot $snapshot_path"

                ./scripts/run.py --snapshot "$snapshot_path" \
                                --scene "$original_scene_name" \
                                --test_transforms \
                                --training_mode=RFL \
                                --identifier scene-training_mode \
                                --folder "$folder" \
                                --no-gui

                if [ $? -ne 0 ]; then
                    echo "Error occurred PSNR evaluation $scene_name."
                else
                    echo "PSNR evaluated for $scene_name."
                fi
            else
                echo "No snapshot found for $scene_name. Skipping..."
            fi
        fi
    done
fi

echo "All scenes processed."
