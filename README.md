# SC3000 Artificial Intelligence
> Lab 1, Lab Group: Z59, AY2022/23 S2

## Team Members
- Budi Syahiddin
- Faiz Rosli
- Chin Tao

## Problem: CartPole
Make an agent that can balance the pole on the cart. For more information on the environment, check out 
[OpenAI Gym](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Chosen Solution
Q-Learning. Why? CartPole is a simple problem, does not have many states and there are only 2 possible actions! 
Also, Q-Learning is very easy to implement from first scratch and does not require a strong computer to train the agent.
Computing Q-Table state-action pair can be described using the equation below
$$
Q(S, a) = Q(S, a) + \alpha [R_{t+1} + \gamma \overbrace{\max_{a \in A(S_{t+1})}(Q(S_{t+1}, a))}^{\text{take best future action}} - Q(S,a))]
$$

## Chosen Policy
Epsilon Greedy. Why? Similar to the previous reason, it is simple to implement.
