---
layout: post
title:  "Finite Markov Decision Processes"
date:   2024-01-07 23:06:32 +0800
categories: RL
permalink: /RL/1
button_text: "RL"
button_path: "/notes"
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


#### 1. The Agent–Environment Interface

**Markov property:** The state that has the Markov property includes information about all aspects of the past agent-environment interaction that make a difference for the future. The MDP framework is an abstraction of the problem of goal-directed learning from interaction.

**State, action, reward chain:** 
$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3$$
 (Define receiving reward to be the very first thing at each time).

In finite MDP, 
$$R_t$$
 and 
$$S_t$$
 have a well-defined probability distribution only on preceding state and actions:

$$p(s', r | s, a) = \Pr(S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a)$$

and they sum to 1: 
$$\sum_{s'\in S}\sum_{r\in R}p(s',r|s,a) = 1$$
, for all 
$$s \in S$$
, 
$$a \in A$$
.

From the four-argument dynamics function 
$$p$$
, we can write the following:

$$p(s'|s,a) = \Pr(S_t=s'|S_{t-1}=s,A_{t-1}=a) = \sum_{r\in R}p(s',r|s,a)$$


$$r(s,a) = \mathbb{E}(R_t|S_{t-1}=s,A_{t-1}=a) = \sum_{r\in R}r * p(r|s,a)= \sum_{r\in R}r\sum_{s'\in S}p(s',r|s,a)$$


$$r(s,a,s') = \mathbb{E}(R_t|S_{t-1}=s,A_{t-1}=a, S_{t}=s') = \sum_{r\in R}r * p(r|s,a,s')=\sum_{r\in R}r\frac{p(s',r|s,a)}{p(s'|s,a)}$$


#### 2. Goals, Rewards, Policies, and Value Functions

**Reward:** A simple number 
$$R_t \in \mathbb{R}$$
.

**Return:** The sum of rewards can be defined as:

$$G_t = R_{t+1} + R_{t+2} + ... + R_{T}$$

or

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum^{\infty}_{k=0} \gamma^k R_{t+k+1}, \quad 0 \leq \gamma \leq 1
$$


$$\gamma$$
 is the discount rate. Note 
$$G_t$$
 is the future return (does not include reward at 
$$t$$
). It can also be written as:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} +... = R_{t+1} + \gamma ( R_{t+2} + \gamma R_{t+3} +...) = R_{t+1} + \gamma G_{t+1}, \quad t < T$$


Note that although the return is a sum of an infinite number of terms, it is still finite if the series of possible rewards are bounded, and 
$$\gamma < 1$$
:

$$G_t \leq \sum_{k=0}^{\infty}\gamma^k * R_{\max} = \frac{1}{1-\gamma}R_{\max}$$


**Policy:** A mapping from states to probabilities of selecting each possible action: 
$$\pi(a|s) = p(A_t=a | S_t=s )$$
.

**Value function:** The expected return when starting in 
$$s$$
 and following 
$$\pi$$
 thereafter, 
$$v_{\pi}(s)$$
. For MDPs, we can define 
$$v_{\pi}$$
 to be:

$$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s] = \mathbb{E}_{\pi}[\sum^{\infty}_{k=0}\gamma^k R_{t+k+1} | S_t = s], \quad \forall s \in S $$


**Action-value function:** Define the value of taking action 
$$a$$
 in state 
$$s$$
 under a policy 
$$\pi$$
, denoted 
$$q_{\pi}(s,a)$$
, as the expected return starting from 
$$s$$
, taking action 
$$a$$
, and thereafter following policy 
$$\pi$$
:

$$q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = A] = \mathbb{E}_{\pi}[\sum^{\infty}_{k=0}\gamma^k R_{t+k+1} | S_t = s, A_t = s], \quad \forall s \in S$$


A fundamental property of value functions used in RL and DP is that they satisfy recursive relationships: For any policy 
$$\pi$$
 and any state 
$$s$$
, the following `consistency condition` holds between the value of a state and its possible successor states:

$$ v_{\pi}(s) $$
$$ = \mathbb{E}_{\pi}[G_t | S_t = s] $$
$$ = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s] $$
$$ = \sum_a\pi(a|s)\sum_{s'}\sum_r p(s',r|s,a)[r + \gamma \mathbb{E}[G_{t+1}|S_{t+1} = s'] ] $$
$$ = \sum_a\pi(a|s)\sum_{s', r}p(s',r|s,a)[r + \gamma v_{\pi}(s')], \quad \forall s \in S $$


Note the above equation is the `Bellman Equation` for 
$$v_{\pi}$$
.


#### 3. Optimal Policies and Optimal Value Functions

**Optimal Policy:** A better policy satisfies 
$$v_{\pi}(s) \geq v_{\pi'}(s) ,\forall s \in S$$
. The optimal policy is the one that is better than or equal to all other policies:

$$v_{*}(s)=\max_{\pi}v_{\pi}(s), \quad \forall s \in S$$

Optimal policies also share the same optimal action-state function:

$$q_*(s,a) = \max_{\pi}q_{\pi}(s,a), \quad \forall s \in S, a\in A$$

For the state-action pair 
$$(s,a)$$
, this function gives the expected return for taking action 
$$a$$
 in state 
$$s$$
 and thereafter following an optimal policy, thus:

$$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t=a]$$


Because 
$$v_*$$
 is the value function for a policy, it must satisfy the `consistency condition` given by the `Bellman equation` for state values. Because it is the optimal value function, however, 
$$v_*’s$$
 consistency condition can be written in a special form without reference to any specific policy. This is the Bellman equation for 
$$v_*$$
, or the `Bellman optimality equation`:

$$v_{*}(s) = \max_{a\in A(s)} q_{\pi_*}(s,a)$$


$$v_{*}(s) = \max_a\mathbb{E}_{\pi_*}[G_t | S_t = s, A_t=a]$$


$$v_{*}(s) = \max_a \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t=a]$$


$$v_{*}(s) = \max_a\mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t=a]$$


$$v_{*}(s) = \max_a\sum_{s', r}p(s',r|s,a)[r + \gamma v_{\pi_*}(s')], \quad \forall s \in S$$


Intuitively, the `Bellman optimality equation` expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state.

The `Bellman optimality equation` for 
$$q_*$$
 is:


$$q_{*}(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t=a]$$


$$q_{*}(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'\in A(S_{t+1})} q_{\pi_*}(S_{t+1},a')(S_{t+1}) | S_t = s, A_t=a]$$


$$q_{*}(s,a) = \sum_{s', r}p(s',r|s,a)[r + \gamma \max_{a'\in A(S_{t+1})} q_{\pi_*}(s', a')], \quad \forall s \in S, a \in A(s) $$


**Solution:** For finite MDPs, the `Bellman optimality equation` for 
$$v_*$$
 has a unique solution: It is a system of equations, one for each state. Only the values 
$$v_*(s)$$
 are unknown if the dynamics 
$$p$$
 of the environment are known.

Once one has 
$$v_*$$
, to determine an optimal policy, for each state 
$$s$$
, there will be one or more actions at which the max is obtained in the `Bellman Optimality equation`. Any policy that only assigns probability to such actions is optimal (One step search, as 
$$v_*$$
 already takes into account the reward consequences of all possible future behavior).

If one has 
$$q_*$$
, the agent can simply find any action that maximizes 
$$q_*(s,a)$$
.
