This branch is the code of TD3 with a uniformly random behavior policy for the paper

*Generalized Off-Policy Actor-Critic* \
Shangtong Zhang, Wendelin Boehmer, Shimon Whiteson (NeurIPS 2019)

*Provably Convergent Two-Timescale Off-Policy Actor-Critic with Function Approximation* \
Shangtong Zhang, Bo Liu, Hengshuai Yao, Shimon Whiteson (ICML 2020)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── agent/TD3_agent.py                              # A random TD3 agent
    └── template_jobs.py.                               # Start random TD3 baseline

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.