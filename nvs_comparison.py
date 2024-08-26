# This script runs the MipNeRF experiments for novel view synthesis (NVS)
# comparison without any regularization (e.g., Laplacian smoothing). It also
# saves optimization results, statistics, and generates plots and tables.

import subprocess
from tabulate import tabulate
from datetime import datetime
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import subprocess

###########################
### CONFIGURATION START ###
###########################

scenes = [
    'bicycle',
    'bonsai',
    'counter',
    'garden',
    'kitchen',
    'room',
    'stump',
    'flowers',
    'treehill'
]
# scenes = ['garden']

methods = ['RFL', 'RFLrelax', 'NeRF']

equal_time = False
iters = 35000   # 35000 as default
seconds = 1200  # 1200 for timeout

###########################
### CONFIGURATION END  ####
###########################

def execute(command):
    try:
        print(f"\033[1;32;40m>> {command}\033[0m")
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\033[1;31;40m{e}\033[0m")
        print(f"\033[1;31;40m{e.stderr}\033[0m")
        raise e


def run_mipnerf():
    for scene in scenes:
        for method in methods:
            id = f"{scene}_{method}_{datetime.now().isoformat(timespec='seconds')}"
            id = id.replace('-', '_')
            command = f"""
                ./scripts/run.py \
                    --scene {scene} \
                    --no-gui \
                    --{'n_steps ' + str(iters) if not equal_time else 'seconds ' + str(seconds)}\
                    --save_snapshot \
                    --save_losses \
                    --save_avg_samples_per_ray \
                    --training_mode={method} \
                    --laplacian_mode=Disabled \
                    --identifier {id} \
                    --folder exp/nvs \
            """

            # if method != 'NeRF':
            #     command += ' --random_dropout'  # Not used in paper

            if method != 'NeRF' or scene == 'stump':
                command += ' --early_density_suppression'

            execute(command)

            command = f"""
                ./scripts/run.py \
                --snapshot out/exp/nvs/{id}/model.ingp \
                --scene {scene} \
                --no-gui \
                --test_transforms \
                --training_mode={method} \
                --identifier {id} \
                --folder exp/nvs \
            """
            execute(command)

            print(f"Done with \"{scene}\".\n")

if __name__ == '__main__':
    run_mipnerf()

    folder = './out/exp/nvs'
    summary_folder_name = f"{datetime.now().isoformat(timespec='seconds')}"
    dirs = [name for name in os.listdir(folder) if os.path.isdir(folder + '/' + name)]
    dirs = sorted(dirs)

    # Make sure {folder}/results/{summary_folder_name} exists
    command = f"mkdir -p {folder}/results/{summary_folder_name}"
    execute(command)

    rows = []
    for scene in scenes[:]:
        for method in methods:
            # Use more precise matching to avoid RFL matching RFLrelax
            scene_dirs = [name for name in dirs if name.startswith(f"{scene}_{method}_")]
            if not scene_dirs:
                print(f"Warning: No results found for {scene}_{method}, skipping...")
                continue
            latest = scene_dirs[-1]

            # Only load surface rendering results for RFL method
            if method == 'RFL':
                psnr_ssim_surface = np.load(folder + '/' + latest + '/psnr_ssim_surface_rendering_True.npz')
                psnr_surface = psnr_ssim_surface['psnr']
                ssim_surface = psnr_ssim_surface['ssim']

            # Volume rendering results are available for all methods
            psnr_ssim_volume = np.load(folder + '/' + latest + '/psnr_ssim_surface_rendering_False.npz')
            psnr_volume = psnr_ssim_volume['psnr']
            ssim_volume = psnr_ssim_volume['ssim']

            losses = np.load(folder + '/' + latest + '/losses.npz')
            traintime = losses['time'][-1]

            row = {
                'scene': scene,
                'Method': method,
                'PSNR_volume':  '{0:.2f}'.format(float(psnr_volume)),
                'SSIM_volume':  '{0:.3f}'.format(float(ssim_volume)),
                'traintime': '{0:.1f}'.format(float(traintime))
            }

            if 'RFL' in methods:
                row['PSNR_surface'] = '{0:.2f}'.format(float(psnr_surface)) if method == 'RFL' else '-'
                row['SSIM_surface'] = '{0:.3f}'.format(float(ssim_surface)) if method == 'RFL' else '-'

            rows.append(row)


    df = pd.DataFrame(rows)
    print(df)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    ax.axis('off')
    ax.axis('tight')

    df = df.drop(columns=['SSIM_volume'])
    if 'RFL' in methods:
        df = df.drop(columns=['SSIM_surface'])

    round_dict = {
        'PSNR_volume':  2,
        'SSIM_volume': 2,
        'traintime': 2
    }
    if 'RFL' in methods:
        round_dict['PSNR_surface'] = 2
        round_dict['SSIM_surface'] = 2
    df = df.round(round_dict)

    latex_table = tabulate(df, df.columns, tablefmt="latex_booktabs")
    print(latex_table)
    latex_document = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \usepackage[table]{xcolor}
    \begin{document}
    \rowcolors{2}{gray!25}{white}
    %s
    \end{document}
    """ % latex_table
    # Save LaTeX to a .tex file
    with open(f'table_mipnerf_{seconds if equal_time else iters}.tex', "w") as f:
        f.write(latex_document)

    # Convert the LaTeX file to PDF using pdflatex
    subprocess.run(["pdflatex", f'table_mipnerf_{seconds if equal_time else iters}.tex'])

    # Move to the results folder
    command = f"mv table_mipnerf_{seconds if equal_time else iters}.pdf {folder}/results/{summary_folder_name}/"
    execute(command)

    fig = plt.figure(figsize=(10, 20))
    fig.tight_layout()
    ax = fig.add_subplot(len(scenes), 1, 1)

    idx = 1
    for scene in scenes[:]:
        if 'RFL' in methods:
            scene_mw_dirs = [name for name in dirs if name.startswith(f"{scene}_RFL_")]
            if scene_mw_dirs:
                latest_mw = scene_mw_dirs[-1]
                avg_spr_mw = np.load(folder + '/' + latest_mw + '/avg_samples_per_ray.npz')
        if 'NeRF' in methods:
            scene_nerf_dirs = [name for name in dirs if name.startswith(f"{scene}_NeRF_")]
            if scene_nerf_dirs:
                latest_nerf = scene_nerf_dirs[-1]
                avg_spr_nerf = np.load(folder + '/' + latest_nerf + '/avg_samples_per_ray.npz')
        if 'RFLrelax' in methods:
            scene_rflnerf_dirs = [name for name in dirs if name.startswith(f"{scene}_RFLrelax_")]
            if scene_rflnerf_dirs:
                latest_rflnerf = scene_rflnerf_dirs[-1]
                avg_spr_rflnerf = np.load(folder + '/' + latest_rflnerf + '/avg_samples_per_ray.npz')

        ax = fig.add_subplot(len(scenes), 1, idx)
        ax.set_title('Avg. samples per ray for "' + scene + '" (log scale)')
        ax.set_yscale('log')
        if 'RFL' in methods and 'latest_mw' in locals():
            ax.plot(avg_spr_mw['time'], avg_spr_mw['avg_spr'], label='RFL')
        if 'NeRF' in methods and 'latest_nerf' in locals():
            ax.plot(avg_spr_nerf['time'], avg_spr_nerf['avg_spr'], label='NeRF')
        if 'RFLrelax' in methods and 'latest_rflnerf' in locals():
            ax.plot(avg_spr_rflnerf['time'], avg_spr_rflnerf['avg_spr'], label='RFLrelax')
        ax.legend()

        idx += 1

    fig.subplots_adjust(wspace=0.6, hspace=0.6)
    fig.savefig(f'{folder}/results/{summary_folder_name}/avg_spr_mipnerf_{seconds if equal_time else iters}.pdf')

    fig = plt.figure(figsize=(10, 20))
    fig.tight_layout()
    ax = fig.add_subplot(len(scenes), 1, 1)
    idx = 1
    for scene in scenes[:]:
        if 'RFL' in methods:
            scene_mw_dirs = [name for name in dirs if name.startswith(f"{scene}_RFL_")]
            if scene_mw_dirs:
                latest_mw = scene_mw_dirs[-1]
                losses_mw = np.load(folder + '/' + latest_mw + '/losses.npz')
        if 'NeRF' in methods:
            scene_nerf_dirs = [name for name in dirs if name.startswith(f"{scene}_NeRF_")]
            if scene_nerf_dirs:
                latest_nerf = scene_nerf_dirs[-1]
                losses_nerf = np.load(folder + '/' + latest_nerf + '/losses.npz')
        if 'RFLrelax' in methods:
            scene_rflnerf_dirs = [name for name in dirs if name.startswith(f"{scene}_RFLrelax_")]
            if scene_rflnerf_dirs:
                latest_rflnerf = scene_rflnerf_dirs[-1]
                losses_rflnerf = np.load(folder + '/' + latest_rflnerf + '/losses.npz')

        ax = fig.add_subplot(len(scenes), 1, idx)
        ax.set_yscale('log')
        ax.set_title('Log-loss for "' + scene  + '" (Huber)')
        if 'RFL' in methods and 'latest_mw' in locals():
            ax.plot(losses_mw['time'], losses_mw['losses'], label='RFL')
        if 'NeRF' in methods and 'latest_nerf' in locals():
            ax.plot(losses_nerf['time'], losses_nerf['losses'], label='NeRF')
        if 'RFLrelax' in methods and 'latest_rflnerf' in locals():
            ax.plot(losses_rflnerf['time'], losses_rflnerf['losses'], label='RFLrelax')
        ax.legend()

        idx += 1

    fig.subplots_adjust(wspace=0.6, hspace=0.6)
    fig.savefig(f'{folder}/results/{summary_folder_name}/losses_mipnerf_{seconds if equal_time else iters}.pdf')
