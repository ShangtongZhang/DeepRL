This branch is the code for the paper

*Breaking the Deadly Triad with a Target Network* \
Shangtong Zhang, Hengshuai Yao, Shimon Whiteson (ICML 2021)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── baird                                       # Entrance of Baird's counterexample experiments 
    ├── deep_rl/agent/TargetNet_agent.py                # Q-learning with target network 
    └── template_plot.py                                # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.