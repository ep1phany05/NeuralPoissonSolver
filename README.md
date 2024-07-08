# Neural Poisson Solver: A Universal and Continuous Framework for Natural Signal Blending
PyTorch implementation of Neural Poisson Solver.

## Pipeline
<img src='assets/pipeline.png' alt="pipeline"/>

## Setup
We provide a conda environment setup file including all of the above dependencies. Create the conda environment Neural Poisson Solver by running:
```
conda create -n neural-poisson-solver python=3.8
conda activate neural-poisson-solver
pip install -r requirements.txt
```
 
## Running

### 2D scene
For 2D scene blending tasks, we employ [DINER](https://github.com/Ezio77/DINER) as the backbone network. 

#### Data preparation
You need to prepare the following data and place them in the `data/2d/` folder. Each blending scene should include:
- `src.pth`: Source scene's INR model.
- `tgt.pth`: Target scene's INR model.
- `roi.png`: Blending region.
- `cfg.npy`: Blending center coordinates.

The directory structure should look like this:
```
data/2d/
├── scene_1/
│   ├── src.pth          # Source scene's INR model
│   ├── tgt.pth          # Target scene's INR model
│   ├── roi.png          # Blending region
│   ├── cfg.npy          # Blending center coordinates
└── ...
```

#### Scene blending
```bash
export PYTHONPATH=$(pwd)
python src/blending/blend_2d.py --save_dir results/2d/scene_1/ --root_dir data/2d/scene_1/ --use_numpy False
```

### 3D scene
For 3D scene blending tasks, we employ [NeRF](https://github.com/yenchenlin/nerf-pytorch) as the backbone network.

#### Data preparation
You need to prepare the following data and place them in the `data/3d/` folder. Each blending scene should include:
```
TODO
```

#### Scene blending
```
TODO
```

## Citation
```
TODO
```