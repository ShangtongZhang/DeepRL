This branch is the code for the paper

*ACE: An Actor Ensemble Algorithm for Continuous Control with Tree Search* \
Shangtong Zhang, Hao Chen, Hengshuai Yao (AAAI 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── plan-ddpg.py                                    # Entrance for the Roboschool experiments
    |   ├── plan_ddpg                                   # Entrance of ACE 
    ├── deep_rl/agent/PlanDDPG_agent.py.py              # Implementation of ACE 
    └── plot_plan.py                                    # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.
