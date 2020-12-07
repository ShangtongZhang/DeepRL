This branch is the code for the paper

*Mean-Variance Policy Iteration for Risk-Averse Reinforcement Learning* \
Shangtong Zhang, Bo Liu, Shimon Whiteson (AAAI 2021)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── mvpi_td3_continuous                         # MVPI-TD3 / TD3 calling
    |   ├── var_ppo_continuous                          # TRVO calling
    |   ├── mvp_continuous                              # MVP calling
    |   ├── risk_a2c_continuous                         # Prashanth baseline calling
    |   ├── tamar_continuous                            # Tamar baseline calling
    |   ├── off_policy_mvpi                             # Offline MVPI calling
    ├── deep_rl/agent/MVPITD3_agent.py                  # MVPI-TD3 / TD3 implementation 
    ├── deep_rl/agent/VarPPO_agent.py                   # TRVO implementation 
    ├── deep_rl/agent/MVP_agent.py                      # MVP implementation 
    ├── deep_rl/agent/RiskA2C_agent.py                  # Prashanth baseline implementation 
    ├── deep_rl/agent/Tamar_agent.py                    # Tamar baseline implementation 
    ├── deep_rl/agent/OffPolicyMVPI_agent.py            # Offline MVPI implementation 
    └── template_plot.py                                # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.