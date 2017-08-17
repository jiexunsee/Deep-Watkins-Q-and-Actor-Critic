# Deep TD Lambda
## Introduction
This repository contains a reinforcement learning agent that uses the TD(lambda) algorithm to solve OpenAI gym games. Many thanks to Siraj's [video](https://www.youtube.com/watch?v=79pmNdyxEGo) for the challenge.

### About TD(lambda)
*TD(lambda) is one of the oldest and most widely used algorithms in reinforcement learning. - Sutton et al., Reinforcement Learning: An Introduction*

TD stands for temporal difference, probably the most important idea to reinforcement learning. One key thing about TD methods is that they bootstrap - they learn estimates based on other estimates. The advantage they have over Dynamic Programming methods is that they do not require a model of the environment. And as compared to Monte Carlo methods, they can be implemented on-line, without having to wait until the end of an episode to learn.

TD(lambda) is a formulation of TD that unifies Monte Carlo and TD methods. It makes use of the mechanism of an eligibility trace. I shall let Sutton and Barto do the explaining as they describe it very elegantly.

> What eligibility traces offer beyond these is an elegant algorithmic mechanism with significant computational advantages. The mechanism is a short term memory vector, the *eligibility trace* e(t), that parallels the long-term weight vector theta(t). The rough idea is that when a component of theta(t) participates in producing an estimated value, then the corresponding component of e(t) is bumped up and then begins to fade away. Learning will occur in that component of theta(t) if a nonzero TD error occurs before the trace falls back to zero. The trace-decay parameter lambda determines the rate at which the trace falls.

## The implementation
I have coded out the TD(lambda) algorithm based on the textbook mentioned above. (It is an excellent book, do check it out if you haven't).

In addition, (raw) TensorFlow is used as it proves to be extremely handy in providing the gradients of the weights to be updated. This is necessary as we need to scale the gradients by the eligibility trace before performing the update.

Using TensorFlow also means that this code can easily be extended to deeper Q learners that, for example, use convolutional networks to estimate value from pixels (like in universe) instead of from numerical states (as in gym). This is also something I will be doing next.

Currently, this learning agent has a vanilla neural network that takes the state as input and outputs a list of Q values for each action. Q learning with TD(lambda) is performed using these estimated values.

## How to run
Simply use `python main.py`. This file is also where you can specify settings like episodes, and the parameters of the agent.

The agent is found in `DeepTDLambdaLearner.py`, where the TD(lambda) algorithm is implemented with the help of TensorFlow.

## Requirements
* tensorflow
* gym
* numpy
