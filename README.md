# Offline RL Algorithms


## Running the code

```
python run_agent.py --env <env_name>  --seed <seed_no>  --exp_name <experiment name> --algorithm <offline rl algorithm>
```

## List of Implemented Algorithms
* 'SAC' : Soft Actor Critic
* 'CQL' : Conservative Q learning
* 'CWR-exp' : Critic weighted regression (exponential filtering) or AWAC (Advantage weighted actor critic)
* 'CWR-binary': Critic weighted regression (binary filtering)
* 'CWR-binary-max': Critic weighted regression (binary filtering with pessimistic advantage estimates)


## Plotting
```
python plot.py <data_folder> --value <y_axis_coordinate> 
```

The plotting script will plot all the subfolders inside the given folder. The value is the y-axis that is required.
'value' can be:
* AverageTestEpRet

