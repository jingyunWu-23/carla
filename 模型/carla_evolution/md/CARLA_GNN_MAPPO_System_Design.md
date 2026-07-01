
# CARLA Multi-Agent GNN Risk + MAPPO System Design

## 1. System Overview
Goal:
- Fix ego mis-stopping
- Improve MAPPO coordination
- Prevent agent collision
- Enable diverse adversarial scenarios

Core idea:
Graph-based interaction modeling + latent adversarial modes + constrained optimization.

---

## 2. System Architecture

CARLA state
→ Interaction Graph
→ GNN Encoder
→ (Ego Branch + MAPPO Branch)

---

## 3. Ego (Main Vehicle)

### 3.1 Input
- Ego-centered interaction graph
- Vehicle states: position, velocity, lane, heading

### 3.2 GNN Risk Model
risk_ego = GNN(G_ego)

### 3.3 Adaptive Threshold τ
τ = τ0 + k1*traffic_density + k2*ego_speed + k3*uncertainty

Recommended:
τ0=0.5, k1=0.3, k2=0.2, k3=0.4

### 3.4 Ego Reward
R_ego = R_progress + α*R_speed - β*max(0, risk_ego - τ)

Recommended:
α=0.1~0.3
β=1~3

---

## 4. MAPPO Adversarial Vehicles

### 4.1 Latent Interaction Mode
z = GNN(graph_state)

Modes:
- z1: front pressure
- z2: lateral squeeze
- z3: rear pressure
- z4: coordinated trap

Policy:
π(a | s, z)

---

### 4.2 Role-conditioned Policy
π_i(a | s, z, role_i)

Roles:
- front attacker
- side blocker
- rear pressure
- coordinator

---

### 4.3 MAPPO Reward
R_adv =
Δrisk_ego + λ1*crash_bonus - λ2*collision_penalty - λ3*formation_violation

Recommended:
λ1=5~10
λ2=2~5
λ3=1~3

---

## 5. GNN Interaction Model

Graph:
- Nodes: vehicles
- Edges: TTC < threshold or distance < threshold

TTC threshold: 3–5s
Distance threshold: 30–50m

Outputs:
- node embeddings
- risk scores
- latent mode z

---

## 6. Formation Constraint
L_form = Σ(||x_i - x_j|| - d_ij*)²

Prevents:
- internal collisions
- formation collapse

---

## 7. Lagrangian PPO

L = R_ego - λ*(risk - τ)

λ update:
λ ← λ + η(risk - τ)

η = 0.01–0.05

---

## 8. Training Pipeline

Phase 1: GNN pretrain (heuristic imitation)
Phase 2: hybrid risk (50/50)
Phase 3: full system

---

## 9. Problem Mapping

Ego mis-stop → GNN + τ fix
MAPPO collapse → latent mode z
collision → formation constraint
mode collapse → entropy over z

---

## 10. Key Insight

GNN = interaction understanding
τ = safety boundary
Lagrangian PPO = constrained optimization
MAPPO = adversarial generator
