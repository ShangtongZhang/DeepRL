This branch is the code for QUOTA with continuous action in the paper

*QUOTA: The Quantile Option Architecture for Reinforcement Learning* \
Shangtong Zhang, Borislav Mavrin, Linglong Kong, Bo Liu, Hengshuai Yao (AAAI 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── dist-ddpg.py                                    # Entrance for the Roboschool experiments
    |   ├── option_ddpg_continuous                      # Entrance of QUOTA
    ├── deep_rl/agent/QuantileOptionDDPG_agent.py       # Implementation of QUOTA with continuous action
    ├── deep_rl/agent/QuantileDDPG_agent.py             # Implementation of QR-DDPG 
    └── plot_dist-ddpg.py                               # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.
