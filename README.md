This branch is the code for the paper

*Learning Retrospective Knowledge with Reverse Reinforcement Learning* \
Shangtong Zhang, Vivek Veeriah, Shimon Whiteson (NeurIPS 2020)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── reverse_TD_robot                            # The effect of $\lambda$ in Reverse TD 
    |   ├── reverse_TD_robot_tasks                      # Anomaly detection in the microdrone example 
    |   ├── continuous_reverse_TD_tasks                 # Anomaly detection in Reacher-v2 
    ├── deep_rl/agent/ReverseTD.py                      # Reverse TD with discrete action 
    ├── deep_rl/agent/ContinuousReverseTD.py            # Reverse TD with continuous action 
    └── template_plot.py                                # Plotting

The code for Atari experiments is not included in this repo.

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.