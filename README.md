# Deep TD Lambda
## Introduction
This repository contains a reinforcement learning agent that uses the TD(lambda) algorithm to solve OpenAI gym games. Many thanks to Siraj's [video](https://www.youtube.com/watch?v=79pmNdyxEGo) for the challenge.

### About TD(lambda)
*TD(lambda) is one of the oldest and most widely used algorithms in reinforcement learning. - Sutton et al., Reinforcement Learning: An Introduction*

TD stands for temporal difference, which is probably the most important idea to reinforcement learning. One key thing about TD methods is that they bootstrap - they learn estimates based on other estimates. The advantage they have over Dynamic Programming methods is that they do not require a model of the environment. And as compared to Monte Carlo methods, they can be implemented on-line, without having to wait until the end of an episode to learn.

TD(lambda) is a formulation of TD that unifies Monte Carlo and TD methods. It makes use of the mechanism of an eligibility trace. Sutton's textbook explains:

> What eligibility traces offer beyond these is an elegant algorithmic mechanism with significant computational advantages. The mechanism is a short term memory vector, the *eligibility trace* e(t), that parallels the long-term weight vector theta(t). The rough idea is that when a component of theta(t) participates in producing an estimated value, then the corresponding component of e(t) is bumped up and then begins to fade away. Learning will occur in that component of theta(t) if a nonzero TD error occurs before the trace falls back to zero. The trace-decay parameter lambda determines the rate at which the trace falls.

## How to run the code
Run `python main.py`.

The agent is found in `DeepTDLambdaLearner.py`, where the TD(lambda) algorithm is implemented with the help of TensorFlow.

## Requirements
* tensorflow
* gym
* numpy

## The implementation
I have coded out the TD(lambda) algorithm based on the textbook mentioned above.

In addition, raw TensorFlow is used as it proves to be extremely handy in providing the gradients of the weights to be updated. This is necessary as we need to scale the gradients by the eligibility trace before performing the update.

Using TensorFlow also means that this code can easily be extended to deeper Q learners that, for example, use convolutional networks to estimate value from pixels (like in universe) instead of from numerical states (as in gym). Will try this soon.

This learning agent has a vanilla neural network that takes the state as input and outputs a list of Q values for each action. Q learning with TD(lambda) is performed using these estimated values.

***Update**: The algorithm I have used is more formally known as Watkin's Q(lambda) algorithm.*

## Results
Running this algorithm on gym's [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0), we solve the game in less than 100 episodes if the hyperparameters are set right. This is a description of the environment:

> The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

[FrozenLake solution on OpenAI gym](https://gym.openai.com/evaluations/eval_OyMhE4BARAmQDY8ixyZALQ): 85 episodes to solve using Watkin's Q(lambda)

[CartPole solution on OpenAI gym](https://gym.openai.com/evaluations/eval_7wiWpSABTsKdKO3liP3jgw): 195 episodes to solve using Policy Gradient method

## Notes
The stochasticity in the environment introduces an element of luck and messes with the agent's learning. For example, the agent could choose a direction in a state based on the high Q value of that state-action pair, but the randomness could cause it to fall into the water instead. This would cause the agent to learn the a lower Q value for the previous state and action, even if it was a good action to take at that state.

Hence, for a start, I would recommend trying FrozenLake-v0 without the stochasticity in the agent's movement. This allowed me to debug much more easily, as the agent's moves would be fully deterministic. The game would be much more easily solved too. This [link](https://github.com/openai/gym/issues/565) has a simple piece of code to do it. After solving the deterministic FrozenLake, then one could move on to the actual one.

**Updates:** 
* May look to implement [this paper's](http://proceedings.mlr.press/v32/sutton14.pdf) method


