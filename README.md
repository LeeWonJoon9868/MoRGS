# MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes
### [Project Page](https://leewonjoon9868.github.io/MoRGS/) | [Paper](https://arxiv.org/abs/2603.25042)

Official PyTorch implementation for the CVPR 2026 paper: 

"MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes". 

## 👀 Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Rendering and Evaluation](#render-evaluation)
- [Citation](#citation)

## ⚙️ Installation
### Environment Setup

Our software has some submodules, please clone the repo recursively.
```
git clone --recurse-submodules https://github.com/LeeWonJoon9868/MoRGS.git
cd MoRGS
mkdir data logs output
```

We tested with Python 3.10, PyTorch 2.5.1, CUDA 12.1, on a single NVIDIA RTX A5000.

```
conda env create -f environment.yml
conda activate morgs
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/gaussian-rasterization-grad
```


### 3. Pretrained weights (RAFT + SAM2)
| Component | Weight                                                   | Place at |
|-----------|----------------------------------------------------------|----------|
| SEA-RAFT  | `Tartan-C-T-TSKH-spring540x960-M.pth` from [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | `RAFT/models/` |
| SAM 2     | `sam2.1_hiera_large.pt` from [SAM 2](https://github.com/facebookresearch/sam2)                  | `sam2/checkpoints/` |

The default paths are configurable in [`configs/`](configs/) (`flow_model_ckpt`,`sam2_checkpoint`).


## 💾 Data Preparation

We assume datasets are organized as follows:


```
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
```

To generate the `points3D_downsample2.ply`, please use the [multipleviewprogress.sh](https://github.com/hustvl/4DGaussians/blob/master/multipleviewprogress.sh) script from [4DGaussians](https://github.com/hustvl/4DGaussians).

For more information on how datasets are loaded, please see `scene/dataset_readers.py`.

## ⏳ Training
You can train a scene by running:

```
python train.py --config [config_path] -s [source_path] -m [output_name]
```

For example:

```
python train.py --config configs/dynerf.yaml -s data/dynerf/coffee_martini -m ./output/dynerf/coffee_martini
```

Please see specific configuration files in configs for examples, and arguments/__init__.py for the full list of arguments.

<details>
<summary><span style="font-weight: bold;">Useful Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.

  #### --model_path / -m
  Path where the trained model should be stored.

  #### --load_init
  Skip initial-frame fit and load cached checkpoint.

  #### --log_images
  Flag to save rendered images during training.

  #### --log_ply 
  Flag to save point cloud in PLY format during training.


</details>

## 📊 Rendering and Evaluation
###  Rendering
```
python render.py -s <path to scene> -m <path to trained model> # Generate renderings
```
###  Evaluation
```
python metrics_video.py -m <path to trained model> # Compute error metrics on renderings
```

###  Videos
```
python generate_video.py -m <path to trained model> 
```

## Acknowledgements
Our work is partially based on these opening source work: [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [3DGStream](https://github.com/SJoJoK/3DGStream), [QUEEN](https://github.com/NVlabs/queen), [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [SAM2](https://github.com/facebookresearch/sam2).

We appreciate the authors for their contributions.

## ✏️ 📄 Citation
If you find our work useful or interesting, please cite our paper:

```
@article{lee2026morgs,
  title={MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes},
  author={Lee, Wonjoon and Woo, Sungmin and Kim, Donghyeong and Lee, Jungho and Park, Sangheon and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2603.25042},
  year={2026}
}
```
