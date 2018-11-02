This branch is the code for **QUOTA with discrete action** in the paper

[QUOTA: The Quantile Option Architecture for Reinforcement Learning]()

See ```requirements.txt``` and ```Dockerfile``` for dependencies.


```MDP.py``` contains the two chains.
```dist_rl.py``` is the entrance for the Atari game experiments. The function ```batch_atari``` will start QUOTA and baseline algorithms. 
Particularly, the function ```bootstrapped_qr_dqn_pixel_atari``` is the entrance of QUOTA. ```plot_dist_rl.py``` contains functions to generate figures in the paper. Unfortunately I can not upload the raw data for plotting to Github. However I can send it via email upon request.

There is also an ice-cliff-world environment, which is not reported in the paper due to the page limit.

Disclaimer: This branch is based on the DeepRL codebase and is left unchanged after I completed the QUOTA paper. Algorithms other than QUOTA and its baselines are heavily outdated and should never be used.
