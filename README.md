# Offline RL Algorithms

This repository aims to provide a simple PyTorch implementation of state-of-the-art Offline RL methods. Some of the methods like AWAC, CQL, MOPO have been tested on MuJoCo locomotion tasks based on d4rl dataset.


## Installation
* Mujoco
* PyTorch
* d4rl(https://github.com/rail-berkeley/d4rl)
* gym 


As an alternative, to replicate the environment used for running the code (might contain a lot of unnecessary libraries as well):   
```
pip install environment/requirements.txt 
```

or via conda    
```
cd environment
conda create --name <env> --file environment/requirements_conda.txt
```

## Running the code

```
python run_agent.py --env <env_name>  --seed <seed_no>  --exp_name <experiment name> --algorithm <offline rl algorithm>
```

## List of Implemented Algorithms
* 'SAC' : Soft Actor Critic
* 'CQL-rho-fixed' : Conservative Q learning rho-version fixed alpha
* 'CQL-rho-lagrange' : Conservative Q learning rho-version trained alpha
* 'CQL-H-fixed' : Conservative Q learning H-version fixed alpha
* 'CQL-H-lagrange' : Conservative Q learning H-version trained alpha
* 'CWR-exp' : Critic weighted regression (exponential filtering) or AWAC (Advantage weighted actor critic)
* 'CWR-binary': Critic weighted regression (binary filtering)
* 'CWR-binary-max': Critic weighted regression (binary filtering with pessimistic advantage estimates)
* 'EMAQ': Expected Max Q learning operator
* 'MOPO': Model based Offline Policy Optimization




## Algorithm specific details

* MOPO   
MOPO takes two additional arguments:   
```
--lamda_pessimism <value>: the amount of pessimism wrt uncertainty
--rollout_length <value>: length of imagined rollout
```



## Plotting
```
python plot.py <data_folder> --value <y_axis_coordinate> 
```

The plotting script will plot all the subfolders inside the given folder. The value is the y-axis that is required.
'value' can be:
* AverageTestEpRet

