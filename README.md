## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 18.04.3 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.19.2

# Install pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install pip requirements
pip install -r requirements.txt

pip install gym
```


## Env
```bash         
observation = [[robot1.x, robot1.y, robot1.heading, robot2.x, robot2.y, robot2.heading, target.x, target.y], [robot1.x, robot1.y, robot1.heading, robot2.x, robot2.y, robot2.heading, target.x, target.y]]
```

```bash
Hyperparameter:
#env size
room_length=1.0 
room_width=0.5

#obs and visualization circle radius
obs_radius = 0.2

#maximal steps for one episode
termination_step = 2000
```


### Primitive Action 
```bash
Run test_pri.py
```

```bash
                   0              1           2            3
ACTIONLIST = ["move forward", "turn left", "turn right", "stay"]

# move forward: 1 mm per simulation step
# turn: 3 deg per simulation step

action = [action1, action2]
```


### Macro Action 
```bash
Run test_MA.py
```

```bash
# 0 <= x, y <= 1 
ACTIONLIST = [x, y]

# xi, yi: target position of robot i
action = [x1, y1, x2, y2]

# real position in map: 
# x_in_map = x * room_length
# y_in_map = x * room_width

```

