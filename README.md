# Offline RL Algorithms


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

