This branch is the code for the paper

*Deep Residual Reinforcement Learning* \
Shangtong Zhang, Wendelin Boehmer, Shimon Whiteson (AAMAS 2020)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── job.py                                
    |   ├── residual_ddpg_continuous                    # Start Bi-Res-DDPG and its variants 
    |   ├── oracle_ddpg_continuous                      # Start DDPG/Dyna-DDPG/Res-Dyna-DDPG/MVE-DDPG with an oracle model 
    |   ├── model_ddpg_continuous                       # Start DDPG/Dyna-DDPG/Res-Dyna-DDPG/MVE-DDPG with a learned model  
    ├── deep_rl/agent/ResidualDDPG_agent.py             # Implementation of Bi-Res-DDPG and its variants 
    ├── deep_rl/agent/OracleDDPG_agent.py               # Implementation of variants of model-based DDPG with an oracle model 
    ├── deep_rl/agent/ModelDDPG_agent.py                # Implementation of variants of model-based DDPG with a learned model
    └── plot_paper.py                                   # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.