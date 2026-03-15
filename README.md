### VIP-Loco: A Visually Guided Infinite Horizon Planning Framework for Legged Locomotion

---

### Installation

This repository is a modified version of [WMP-Loco](https://wmp-loco.github.io/), modifications roughly include our method, **Warp** backend for depth processing, **JAX** for planning, and additional robot platforms.


#### Prerequisites

- Ubuntu 20.04 / 22.04
- CUDA 12.1+
- Conda

#### 1. Clone the repo

```bash
git clone https://github.com/StochLab/VIP-Loco
cd VIP-Loco
```

#### 2. Create the conda environment

```bash
conda create -n viploco python=3.8
conda activate viploco
```

#### 3. Install dependencies

```bash
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jax[cuda12]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install warp-lang==1.4.1
pip install -r requirements.txt
```

#### 4. Install IsaacGym

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and install it manually inside the conda env:

```bash
cd <isaacgym_path>/python
pip install -e .
```

---

### Usage

#### Training

```bash
conda activate viploco
cd VIP-Loco
python legged_gym/scripts/train.py --headless --sim_device=cuda:0 --wm_device=cuda:0 --task=go1_amp
```

Supported tasks: `go1_amp`, `cassie`, `trona1_w`

#### Evaluation

```bash
# With planner
python legged_gym/scripts/play_plan.py --sim_device=cuda:0 --wm_device=cuda:0 --task=go1_amp --terrain=slope

# Without planner
python legged_gym/scripts/play.py --sim_device=cuda:0 --wm_device=cuda:0 --task=go1_amp --terrain=slope
```

Supported terrains: `slope`, `stair`, `gap`, `climb`, `crawl`, `tilt`

---

### Issues

The code is not in a release version, hence if you face any issues in installation or running the scripts, please feel free to create an issue :)

---

### Acknowledgements

Built on top of [WMP-Loco](https://wmp-loco.github.io/) and [legged_gym](https://github.com/leggedrobotics/legged_gym).