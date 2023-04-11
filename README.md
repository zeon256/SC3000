# SC3000 Artificial Intelligence
> Lab 1, Lab Group: Z59, AY2022/23 S2

## Team Members
- Budi Syahiddin
- Faiz Rosli
- Chin Yueh Tao

## Dependencies
- gym==0.25.2
- gym[classic_control]
- numpy
- matplotlib

## Tested on
- Anaconda + Python 3.10.9
- Windows 10

## Problem: CartPole
Make an agent that can balance the pole on the cart. For more information on the environment, check out 
[OpenAI Gym](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Chosen Solution
Q-Learning. Why? CartPole is a simple problem, does not have many states and there are only 2 possible actions! 
Also, Q-Learning is very easy to implement from first scratch and does not require a strong computer to train the agent.
Computing Q-Table state-action pair can be described using the equation below

$$
Q(s, a) = Q(s, a) + \alpha [R_{t+1} + \gamma \overbrace{\max_{a \in A(s_{t+1})}(Q(s_{t+1}, a))}^{\text{take best future action}} - Q(s,a))]
$$

## Chosen Policy
Epsilon Greedy. Why? Similar to the previous reason, it is simple to implement.

## Discretization
Q-Learning works well for environment with discrete states and actions.
However, CartPole states are continuous! In order to deal with that,
we will need to make the states discrete. The idea is to split the range
of the states into "bins", i.e. intervals. For example, if range is $[1, 10]$, and we let $N = 10$, then each interval is $1$.

## Decaying of $\alpha$ and Exploration Rate
Initially, we tested static $\alpha$ and exploration rate. While it worked after many episodes, the agent took too long to get good. For our decaying function, we chose the following implementation below, with $\text{let}\;y = \text{exploit episode} = 50$ and $\text{let}\;z = \text{episode no}$

$$
\text{Rate} = \max(0.01, \min(1, 1.0 - \log(\text{z} + 1 / y)))
$$

The idea we have is, at the beginning, we want the agent to explore more so that it can explore random state spaces. Then as the number of episode increases, we slowly start to exploit, because by then, the table will already be filled with "good" states and we want the agent to exploit those states
