This branch is the code for the paper

*Average-Reward Off-Policy Policy Evaluation with Function Approximation* \
Shangtong Zhang, Yi Wan, Richard S. Sutton, Shimon Whiteson (ICML 2021)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── linear_ope_boyans_chain                     # Entrance of Boyan's chain experiments 
    |   ├── neural_ope                                  # Entrance of MuJoCo experiments 
    ├── deep_rl/agent/LinearOPEAgent.py                 # GradientDICE / Diff-GQ1 / Diff-GQ2 / Diff-SGQ for Boyan's chain
    ├── deep_rl/agent/NeuralOPEAgent.py                 # GradientDICE / Diff-GQ1 / Diff-GQ2 / Diff-SGQ for MuJoCo  
    └── template_plot.py                                # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.