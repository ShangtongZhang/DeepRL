This branch is the code for **QUOTA with continuous action** in the paper

[QUOTA: The Quantile Option Architecture for Reinforcement Learning](https://arxiv.org/abs/1811.02073)

See ```requirements.txt``` and ```Dockerfile``` for dependencies.

```dist-ddpg.py``` is the entrance for Roboschool experiments. The function ```option_ddpg_continuous``` is the entrance of QUOTA. ```plot_dist-ddpg.py``` contains functions to generate figures in the paper. Unfortunately I can not upload the raw data for plotting to Github. However I can send it via email upon request.

Disclaimer: This branch is based on the DeepRL codebase and is left unchanged after I completed the QUOTA paper. Algorithms other than QUOTA and its baselines are heavily outdated and should never be used.
