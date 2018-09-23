# A Gym environment for the new Arcade Learning Environment (v0.6.0)

**A python Gym environment for the new [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) (v0.6.0) supporting different difficulties and game modes.** Enables experimenting with different Atari game dynamics within the Gym framework. This is fully inspired by the Atari environment in [OpenAI gym](https://github.com/openai/gym). 

## Installation
1. Install the latest Arcade Learning Environment by following the instructions at https://github.com/mgbellemare/Arcade-Learning-Environment.

2. Download a ROM-file for the Atari game, or use a ROM file included in the repository. Title the ROM-file all lowercase, such as 'pong.bin'.

3. Import UpdatedAtariEnv and follow the OpenAI gym API. An introduction to the new ALE is available here: https://arxiv.org/pdf/1709.06009.pdf. 

This wrapper is initially developed and used in [Sample-Efficient Deep RL with Generative Adversarial Tree Search](https://arxiv.org/pdf/1806.05780.pdf).

The following is a list of the available modes and difficulties for each game from the new ALE paper. 

![ALE Modes and Difficulties](https://raw.githubusercontent.com/bclyang/updated-atari-env/master/modes_difficulties.png)
