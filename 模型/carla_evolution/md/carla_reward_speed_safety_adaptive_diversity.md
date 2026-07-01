
# CARLA + MAPPO Reward System (Adaptive τ + Interaction Diversity Final Version)

---

# 1. System Overview

We model CARLA multi-agent interaction as a **Constrained Adversarial MDP**:

- Ego: maximize speed under safety constraint
- MAPPO adversary: generate diverse safety-critical scenarios

---

# 2. Core Principle

We reformulate reward structure as:

> ✔ Speed = objective  
> ✔ Risk = adaptive constraint (not penalty)  
> ✔ Diversity = mandatory exploration pressure

---

# 3. Ego Reward (Final)

\[
R_{ego} =
\alpha \cdot \frac{v}{v_{max}}
+ R_{progress}
- \lambda \cdot risk\_violation
\]

where:

\[
risk\_violation = \max(0, risk - \tau)
\]

---

# 4. Risk Definition

Risk is derived from interaction:

- distance-based proximity
- TTC (if available)
- interaction risk field

---

# 5. Adaptive Risk Threshold τ (Key Contribution)

## 5.1 Full formulation

\[
\tau =
\tau_0
+ \alpha_1 H(Z)
+ \alpha_2 Coverage
+ \alpha_3 EgoPerformance
\]

---

## 5.2 Components

| Term | Meaning |
|------|--------|
| τ0 | base safety threshold |
| H(Z) | interaction mode entropy |
| Coverage | spatial interaction diversity |
| EgoPerformance | adaptive difficulty scaling |

---

## 5.3 Interpretation

- If MAPPO collapses to single attack → τ decreases (harder constraint)
- If MAPPO is diverse → τ increases (more exploration space)
- If ego performs poorly → τ increases (curriculum easing)

---

# 6. Interaction Diversity (MAPPO Core Objective)

## 6.1 Interaction mode definition

We define latent interaction modes:

\[
z \in \{
cut-in, blocking, braking, tail-pressure, multi-agent squeeze
\}
\]

Each trajectory is mapped via:

\[
z = cluster(\phi(s,a))
\]

---

## 6.2 Diversity reward

\[
R_{diversity} = - \log p(z)
\]

or equivalently:

\[
R_{diversity} = H(Z)
\]

---

## 6.3 Effect

Prevents MAPPO from collapsing into:
- single cut-in strategy
- single blocking behavior
- single braking pattern

---

# 7. MAPPO Adversarial Reward (Final)

\[
R_{adv} =
10 \cdot \Delta risk \cdot w(d)
+ R_{crash}
+ \beta \cdot R_{diversity}
\]

---

## 7.1 Risk improvement term

\[
\Delta risk = U_{prev} - U_{curr}
\]

\[
w(d) = \exp(-d / 40)
\]

---

## 7.2 Crash reward

- ego crash: +100
- adversary crash: -40

---

# 8. Risk-Constrained Optimization Form

\[
R_{ego} \ \text{optimized under constraint:} \quad risk \leq \tau
\]

This defines a **CMDP with adaptive constraint boundary**.

---

# 9. Training Dynamics

## Ego:
- learns speed-optimal policy inside adaptive safe envelope

## MAPPO:
- generates multi-modal adversarial scenarios
- must maintain interaction diversity
- cannot collapse to single strategy

---

# 10. Expected System Behavior

### Before:
- low-speed equilibrium
- single-mode adversary
- weak interaction pressure

---

### After:
- diverse adversarial scenarios
- dynamic safety boundary (τ)
- stable speed–safety trade-off
- improved robustness training

---

# 11. Key Insight

This system transforms MAPPO from:

> risk maximizer

into:

> structured scenario generator with adaptive safety boundary and diversity constraint
