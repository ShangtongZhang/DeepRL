This branch is the code for QUOTA with discrete action in the paper

QUOTA: The Quantile Option Architecture for Reinforcement Learning \
Shangtong Zhang, Borislav Mavrin, Linglong Kong, Bo Liu, Hengshuai Yao (AAAI 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── MDP.py                                          # Chain 1 and Chain 2 
    ├── dist_rl.py                                      # Entrance for the Atari game experiments
    |   ├── batch_atari                                 # Start QUOTA and baseline algorithms
    |   ├── bootstrapped_qr_dqn_pixel_atari             # Entrance of QUOTA
    ├── deep_rl/agent/BootstrappedNStepQRDQN_agent.py   # Implementation of QUOTA with discrete action
    └── plot_dist_rl.py                                 # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.
