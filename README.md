This branch is the code for the paper

*Provably Convergent Two-Timescale Off-Policy Actor-Critic with Function Approximation* \
Shangtong Zhang, Bo Liu, Hengshuai Yao, Shimon Whiteson (ICML 2020)

    .
    ├── Dockerfile                            # Dependencies
    ├── requirements.txt                      # Dependencies
    ├── template_jobs.py                      # Entrance of the experiments
    |   ├── gem_baird                         # Entrance of Baird's counterexample experiments 
    |   ├── cof_pac                           # Entrance of Reacher experiments 
    ├── deep_rl/agent/GEM_agent.py            # Implementation of GEM and GEM-ETD for tabular / linear settings 
    ├── deep_rl/agent/COFPAC_agent.py         # Implementation of COF-PAC with semi-gradient emphasis learning 
    └── template_plot.py                      # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.