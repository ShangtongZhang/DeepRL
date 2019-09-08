This branch is the code for the paper

*Generalized Off-Policy Actor-Critic* \
Shangtong Zhang, Wendelin Boehmer, Shimon Whiteson (NeurIPS 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── MDP.py                                          # Two-circle MDP 
    ├── job.py                                          # Entrance for the Mujoco experiments
    |   ├── batch                                       # Start Geoff-PAC and baseline algorithms
    |   ├── geoff_pac                                   # Entrance of Geoff-PAC / ACE / Off-PAC
    ├── deep_rl/agent/GeoffPAC_agent.py                 # Implementation of Geoff-PAC / ACE / Off-PAC
    └── plot_paper.py                                   # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.