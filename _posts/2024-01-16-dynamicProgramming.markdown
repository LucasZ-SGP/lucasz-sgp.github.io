---
layout: post
title:  "Dynamic Programming in RL"
date:   2023-01-16 01:32:00 +0800
categories: RL
permalink: /RL/2
button_text: "RL"
button_path: "/notes"
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


#### 1. Policy Evaluation

Recall for an arbitrary policy $$\pi$$:

$$
\begin{align*}
v_{\pi}(s) &= \mathbb{E}_{\pi}[G_t | S_t = s] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s] \\
&= \sum_a\pi(a|s)\sum_{s', r}p(s',r|s,a)[r + \gamma v_{\pi}(s')], \quad \forall s \in S
\end{align*}
$$

If the environment's dynamics are completely known, then it is a system of $$\lvert S \rvert$$ simultaneous linear equations in $$\lvert S \rvert$$ unknowns. In this case, `Iterative solution` methods are most suitable: consider a sequence of approximate value functions $$v_0, v_1, v_2 ...$$, with the initial approximation $$v_0$$ chosen arbitrarily, except at the terminal state, where it has to be 0, and each successive approximation is obtained by using the `Bellman Equation` for $$v_{\pi}$$ as an update rule:

$$
\begin{align*}
v_{k+1}(s) &= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s] \\
&= \sum_a\pi(a|s)\sum_{s', r}p(s',r|s,a)[r + \gamma v_{k}(s')], \quad \forall s \in S
\end{align*}
$$

 $$v_k = v_{\pi}$$ is a fixed point for this update rule because the `Bellman equation` for $$v_{\pi}$$ assures us of equality in this case. It can be shown that the sequence of $$v_k$$ converges to $$v_{\pi}$$.

`Expected updates` can be performed in place or using two arrays. With the former, new values immediately overwrite the old ones.


#### 2. Policy Improvement

Recall that following an existing policy $$\pi$$, consider selecting $$a$$ in $$s$$ and thereafter following the existing policy $$\pi$$. The value of this way of behaving is:

$$
\begin{align*}
q_{\pi}(s,a) &= \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s,A_t=a] \\
&= \sum_{s',r}p(s',r|s,a)[r + \gamma v_{\pi}(s')]
\end{align*}
$$

If it is better to select such action $$a$$ thereafter follow policy $$\pi$$ than it would be to follow $$\pi$$ all the time, then the new policy would be better overall:

 $$q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s), \forall s \in S$$, then $$\pi'$$ is a better policy, i.e. $$v_{\pi'}(s) \geq v_{\pi}(s)$$

**Proof:**

$$
\begin{align*}
v_{\pi}(s) &\leq q_{\pi}(s, \pi'(s)) \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s, A_t = \pi'(s)] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1})) | S_t = s] && \text{(the condition)} \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma \mathbb{E}_{\pi}[R_{t+2} + \gamma v_{\pi}(S_{t+2}) | S_{t+1}, A_{t+1} = \pi'(S_{t+1})] | S_t = s] && \text{(the action-value function)} \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2}) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_{\pi}(S_{t+3}) | S_t = s] \\
&\ldots \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \ldots | S_t = s] \\
&= v_{\pi'}(s)
\end{align*}
$$

So far we have seen how we can easily evaluate a change in the policy at a single state. Consider changes at all stages, using a greedy policy $$\pi'$$, given by:

$$
\begin{align*}
\pi'(s) &= \text{argmax}_a q_{\pi}(s,a) \\
&= \text{argmax}_a \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s, A_t = a] \\
&= \text{argmax}_a \sum_{s', r}p(s', r|s,a)[r + \gamma v_{\pi}(s')]
\end{align*}
$$

If the new greedy policy, $$\pi'$$, is as good as, but not better than, the old policy $$\pi$$, then:

$$
\begin{align*}
v_{\pi'}(s) &= \max_a \mathbb{E}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s, A_t = a] \\
&= \max_a \sum_{s', r}p(s', r|s,a)[r + \gamma v_{\pi}(s')]
\end{align*}
$$

which is the same as the `Bellman Optimality Equation`, and therefore $$v_{\pi'}$$ must be $$v_*$$. It also applies to stochastic policies where a probability $$\pi(a | s)$$ is given for each possible action at state $$s$$.
#### 3. Policy Iteration


Once a policy, $$\pi$$, has been improved using $$v_{\pi}$$ to yield a better policy, $$\pi'$$, we can then compute $$v_{\pi'}$$ and improve it again to yield an even better $$\pi''$$. A sequence of monotonically improving policies and value functions can be obtained:

$$
\pi_0 \xrightarrow {\text{Evaluation}} v_{\pi_0} \xrightarrow {\text{Improvement}} \pi_1 \xrightarrow {\text{Evaluation}} v_{\pi_1} \xrightarrow {\text{Improvement}} \pi_2 \xrightarrow {\text{Evaluation}} v_{\pi_2} \xrightarrow \dots
$$

Because a finite MDP has only a finite number of deterministic policies, this process must converge to an optimal policy and the optimal value function in a finite number of iterations.

#### 4. Value Iteration

Each of the policy iterations involves policy evaluation, which may itself be an iterative computation requiring multiple sweeps through the state set. In fact, the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One special case is when policy evaluation is stopped after just one sweep. This is called `value iteration`:

$$
\begin{align*}
v_{k+1}(s) &= \max_a \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s, A_t = a] \\
&= \max_a \sum_{s',r} p(s', r |s,a)[r + \gamma v_k(s')]
\end{align*}
$$

For arbitrary $$v_0$$, the sequence ${v_k}$ can be shown to converge to $$v_*$$. Note that `value iteration` is obtained simply by turning the `Bellman optimality equation` into an update rule. Also note how the `value iteration update` is identical to the `policy evaluation update` except that it requires the max to be taken over all actions.
