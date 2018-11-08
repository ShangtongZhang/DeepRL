This branch is the code for **ACE** in the paper

[ACE: An Actor Ensemble Algorithm for Continuous Control with Tree Search](https://arxiv.org/abs/1811.02696)

See ```requirements.txt``` and ```Dockerfile``` for dependencies.

```plan-ddpg.py``` is the entrance for the Roboschool experiments. The function ```plan_ddpg``` is the entrance of ACE. ```plot_ddpg.py``` contains functions to generate figures in the paper. Unfortunately I can not upload the raw data for plotting to Github. However I can send it via email upon request.

Disclaimer: This branch is based on the DeepRL codebase and is left unchanged after I completed the ACE paper. Algorithms other than ACE and its baselines are heavily outdated and should never be used.