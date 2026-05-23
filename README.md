# MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes
### [Project Page]([https://sungmin-woo.github.io/prodepth/](https://leewonjoon9868.github.io/MoRGS/)) | [Paper](https://arxiv.org/pdf/2407.09303)

Official PyTorch implementation for the CVPR 2026 paper: "MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes". 

## 👀 Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Rendering and Evaluation](#render-evaluation)
- [Citation](#citation)

- ## ⚙️ Installation
1. Environment Setup
You can install the dependencies with:
```
git clone --recurse-submodules https://github.com/LeeWonJoon9868/MoRGS.git
cd MoRGS
mkdir data
mkdir logs
mkdir output

conda create -n morgs python=3.11
conda activate morgs
pip install -e .
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/gaussian-rasterization-grad
```
We ran out experiments with PyTorch 2.5.1, CUDA 12.1.

2. Download weights for RAFT and SAM2.
For RAFT, please download their pretrained weights Tartan-C-T-TSKH-spring540x960-M.pth from their official repo(https://github.com/princeton-vl/SEA-RAFT).
For SAM2, please download their pretrained weights 1_hiera_large.pt from their official repo(https://github.com/facebookresearch/sam2).

## 💾 Data Preparation
We assume datasets are organized as follows:
'''
| --- data
|   | [dataset_directory]
│     | [scene_name] 
│   	  | cam01
|            | images
|     		  | ---0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | cam02
|            | images
│     		  | --- 0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | ...
│   	  | sparse_
│     		  | --- cameras.bin
│     		  | --- images.bin
│     		  | --- ...
│   	  | points3D_downsample2.ply
│   	  | poses_bounds.npy
'''
To generate the points3D_downsample2.ply , please use the multipleviewprogress.sh script from 4DGaussians(https://github.com/hustvl/4DGaussians).
For more information on how datasets are loaded, please see scene/dataset_readers.py.

## ⏳ Training
You can train a scene by running:
'''
python train.py --config [config_path] -s [source_path] -m [output_name]
'''
For example:
'''
python train.py --config configs/dynerf.yaml -s data/dynerf/coffee_martini -m ./output/dynerf/coffee_martini
'''
Please see specific configuration files in configs for examples, and arguments/__init__.py for the full list of arguments.
<details>
<summary><span style="font-weight: bold;">Useful Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
</details>
