# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

The objective is to find the optimal policy that enables an agent to reach the goal state in the FrozenLake environment while maximizing the expected cumulative reward. The Value Iteration algorithm repeatedly updates the value of each state using the Bellman Optimality Equation until convergence and then derives the optimal policy from the computed value function.




## Software Requirements
```

Python 3.x

Gymnasium

NumPy

Matplotlib

Google Colab / Jupyter Notebook
```


## Environment Description
```
env_desc = [
    "SFFG",
    "FHFH",
    "FFFH",
    "HFFF"
]
```

The environment is a custom 4×4 FrozenLake grid in which the goal state is placed at the top-right corner instead of the default bottom-right position. The agent begins at the Start (S) state and must safely navigate through frozen cells (F) while avoiding holes (H). Since the environment is slippery, each action has stochastic outcomes. The Value Iteration algorithm computes the optimal state-value function and policy to efficiently reach the relocated goal while maximizing the expected cumulative reward.



## MDP Representation
```
States (S):
16 states representing the cells of the 4×4 FrozenLake grid.

Actions (A):
Left
Down
Right
Up

Transition Probability (P):
The environment is stochastic due to slippery movement. The probability of moving to the intended or
 neighboring states is defined by the environment.

Reward Function (R):
Goal State: +1
All other states: 0
Discount Factor (γ)

γ = 0.99
```


## Theory
Value Iteration is a dynamic programming algorithm used to compute the optimal value function for an MDP. It repeatedly applies the Bellman Optimality Equation to update the value of each state until the change between successive iterations becomes smaller than a predefined threshold. Once the value function converges, the optimal policy is obtained by selecting the action that maximizes the expected future reward for each state. Value Iteration guarantees convergence to the optimal policy for finite Markov Decision Processes when an appropriate discount factor is used.




## Algorithm

1.Initialize the value function V(s)=0 for all states.

2.Set the discount factor (γ) and convergence threshold (θ).

3.For each state, compute the expected value of every possible action.

4.Update the state value using the maximum expected action value (Bellman Optimality Equation).

5.Repeat the updates until the maximum change in state values is less than θ.

6.Extract the optimal policy by selecting the action with the highest expected value for each state.

7.Display the optimal state-value function and the optimal policy.





## Python Program

```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
env = env.unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99
theta = 1e-8

# Value Iteration Algorithm
def value_iteration(env, gamma=0.99, theta=1e-8):

    V = np.zeros(n_states)
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            best_value = np.max(action_values)

            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )
        policy[s] = np.argmax(action_values)
    return V, policy, iteration

# Run Value Iteration
V, policy, iterations = value_iteration(
    env,
    gamma=gamma,
    theta=theta
)

# Display Output
print("Name: Yugabharathi T")
print("Register Number: 212224040375")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()



```

## Output


<img width="475" height="382" alt="image" src="https://github.com/user-attachments/assets/0d016977-bd4b-4d78-8eef-b3541617cd84" />



## Result

The Value Iteration algorithm was successfully implemented on the FrozenLake-v1 (4×4) environment using Gymnasium. The algorithm converged after a finite number of iterations and computed the optimal state-value function and the optimal policy, enabling the agent to maximize the expected cumulative reward.


## Inference
```
1.The Value Iteration algorithm successfully adapted to the custom FrozenLake environment and converged to
  the optimal state-value function despite the modified goal location.
2.The computed optimal policy guided the agent through the safest path to the new goal while avoiding hole
  states and maximizing the expected cumulative reward.

```
