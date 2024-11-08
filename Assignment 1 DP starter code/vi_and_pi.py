### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
    default="Deterministic-4x4-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)


"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary of a nested lists
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

    
    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #



    delta = 0
    first =True
    while delta >= tol or first:
        if first:
            first=False
        delta = 0
        for state in range(nS):

            temp = value_function[state]
            
            action = policy[state]

            p = P[state][action]

            store = 0

            for transition in p:
                prob, next_state, reward, term = transition

                store = store + prob * (reward + gamma * value_function[next_state])

            value_function[state] = store

            delta = max(delta, abs(temp - value_function[state]))

    return value_function


    ############################


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #

    for state in range(nS): # No stopping condition, full sweep of all states

        #temp = policy[state]

        vList = np.zeros(nA) # List of transition values for each state

        for action in range(nA): # Each possible action the agent can take at a state

            p = P[state][action] # transition model

            store = 0 

            for transition in p: # for each possible transition, calculate and sum: probability given reward, state * reward + gamma * value of next state

                store = store + transition[0] * (transition[2] + gamma * value_from_policy[transition[1]])

            vList[action] = store # store sum in vList as a transition value
        
        new_policy[state] = np.argmax(vList) # new policy at that state equal
                                             # new policy will always choose the action that yields greatest value

    return new_policy
    ############################


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

    value_function = np.zeros(nS) # set value function to all 0s
    policy = np.random.randint(0, 3, nS) # set policy to random value

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    iterNum = 0

    value_function = policy_evaluation(P,nS,nA,policy,gamma,tol) 

    stable = False
    # policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

    while stable == False: # If policy changes, not stable: run policy evaluation and policy improvement each step
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        store_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)  
        iterNum += 1
        print(iterNum)


        #print(store_policy==policy)
        if (store_policy == policy).all(): # If policy does not change, policy is stable
            stable = True
        
        policy = store_policy # store policy to check for change


    print(policy)
    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""
    iterNum = 0

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    delta = 0 

    first = True

    while delta >= tol or first:
        if first:
            first = False
        
        delta = 0

        for state in range(nS):

            temp = value_function[state]

            vList = np.zeros(nA)

            for action in range(nA):

                p = P[state][action]

                store = 0

                for transition in p:

                    store = store + transition[0] * (transition[2] + gamma * value_function[transition[1]])

                vList[action] = store

            # combining policy evaluation and policy improvement
                
            value_function[state] = max(vList) # policy evaluation

            policy[state] = np.argmax(vList) # policy improvement

            iterNum += 1

            print(iterNum)

            delta = max(delta, abs(temp - value_function[state]))

    #for state in range(nS):
        #policy[state] = np.argmax()
    
    #policy = get_policy()
            
    print(policy)

    return value_function, policy # returns optimal policy
    ############################


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print(
            "The agent didn't reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)


    # Report #
    '''
Name: Derek Kwon
Due Date: March 3rd 2024, 11:59 PM
Class: DATA 320 Reinforcement Learning

In a deterministic environment, the agent does not need to decide randomly as it has all
the data it needs to predict the best outcome. A deterministic environment will always
produce the same output for a given set of inputs. This results in the agent finding the
shortest path to the end every time without fail, for both policy iteration and value iteration.
The deterministic environment had 4 iterations for policy iteration and 112 iterations for
value iteration.

The policy for the deterministic environment is: [ 1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0 ]

A deterministic policy will always output the same action given a particular state, which
allows the agent to solve the maze and reach the end instantly without any errors.


In a stochastic environment, however, the agent must randomly choose its next action
from its current state. This leads to the agent taking much longer to reach the end and
appearing to be more confused since it may end up going backwards and going back to the
start. Sometimes, the agent may not reach the end at all because it fell into a hole or
exceeded 100 steps. It takes many more iterations of policy and value iteration for an agent
in a stochastic environment to get the same result as an agent in a deterministic
environment. The stochastic environment had 6 iterations for policy iteration and 368
iterations for value iteration.

The policy for the stochastic environment is: [ 0 3 0 3 0 0 0 0 3 1 0 0 0 2 1 0 ]

A stochastic policy will choose an action at random based on the probability distribution of
the possible actions an agent can take at a given state. In the stochastic policy, there are
more 0s and the presence of 3s. 

    '''

