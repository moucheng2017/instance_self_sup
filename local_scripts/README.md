# Simple example use:
1. Edit the config yaml file in configs/linear_eval.yaml.
2. Run the following:
```
bash local_scripts/run_linear_eval_local.sh 
```

# Another example use by over writting the config parameters:
```
EVAL_FROM=/Users/xmc28/Desktop/projects/checkpoints/pseudo_sup/exps/episode_4_0614005423.pth \
DATA_DIR=/Users/xmc28/Desktop/projects/data \
bash local_scripts/run_linear_eval_local.sh
```