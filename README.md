This branch is the code for the paper

*GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values* \
Shangtong Zhang, Bo Liu, Shimon Whiteson (ICML 2020)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── gradient_dice_boyans_chain                  # Entrance of Boyan's chain experiments 
    |   ├── off_policy_evaluation                       # Entrance of Reacher experiments 
    ├── deep_rl/agent/GradientDICEAgent.py              # GradientDICE / DualDICE / GenDICE for Boyan's chain
    ├── deep_rl/agent/OffPolicyEvaluationAgent.py       # GradientDICE / DualDICE / GenDICE for Reacher 
    └── template_plot.py                                # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.