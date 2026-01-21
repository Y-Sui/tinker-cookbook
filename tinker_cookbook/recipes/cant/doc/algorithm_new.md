Here is the comprehensive, rigorous description of the **CANT (Self-Evolution)** framework. It includes precise mathematical definitions, the updated reward logic for small $N$, and strict token-level credit assignment strategies suitable for PPO/DPO training logs.

---

# CANT: Self-Evolution Framework
**Experimental Protocol & Mathematical Formulation**

## 1. Problem Definition & Interaction Protocol
We define a training trajectory $\tau$ involving $N$ agents (e.g., $N=4$) sharing the same policy parameter $\pi_\theta$. Each agent $i$ is assigned a unique **Persona** $P_i$ (e.g., "Skeptic", "Constructive", "Analyst") to enforce diverse initial viewpoints.

The interaction for a single query $q$ is strictly divided into four computational rounds to isolate the causal impact of the critique.

### Round 1: Proposal (Initialization)
*   **State ($s_1$):** User Query $q$ + Agent Persona $P_i$.
*   **Action ($a_1$):** Each Agent $i$ generates an **Initial Solution** $S_i^{init}$.
    *   *Objective:* Generate a diverse set of candidate answers. No evaluation occurs in this step.

### Round 2: Blind Evaluation & Critique (The Baseline)
*   **State ($s_2$):** Query $q$ + Set of all Initial Solutions $\{S_0^{init}, ..., S_{N-1}^{init}\}$.
*   **Action ($a_2$):** Each Agent $i$ generates a response containing two strictly structured segments:
    1.  **Segment B (Blind Ranking):** A structured ranking (e.g., pairwise comparison list) of all peers based *solely* on $S^{init}$.
        *   *Constraint:* Agent $i$ has **not** seen any critiques yet. This establishes the "Pre-Debate Baseline."
    2.  **Segment C (Targeted Critique):** The agent selects a subset of peers to critique.
        *   *Format:* Must use XML tags: `<target>Agent k</target>` followed by the critique text $C_{i \to k}$.
        *   *Content:* Identification of logical fallacies, factual errors, or missing constraints.

### Round 3: Revision (The Outcome)
*   **State ($s_3$):** Query $q$ + All Initial Solutions $\{S^{init}\}$ + All Critiques $\{C_{0 \to \cdot}, ..., C_{N-1 \to \cdot}\}$.
*   **Action ($a_3$):** Each Agent $i$ generates:
    1.  **Segment D (Revision):** A **Revised Solution** $S_i^{rev}$ that incorporates valid feedback or defends against invalid critiques.

### Round 4: Final Verdict (The Outcome)
*   **State ($s_4$):** Query $q$ + All Revised Solutions $\{S^{rev}\}$.
*   **Action ($a_4$):** Each Agent $i$ generates:
    1.  **Segment E (Final Ranking):** A re-evaluation of all peers' *revised* solutions.

---

## 2. Mathematical Formulation of Valuation

To calculate rewards, we must quantify the "Group Consensus Value" of each agent before and after the debate.

**2.1. Pairwise Win Matrix**
Let $w_{j \to k}^t \in \{0, 1\}$ be the binary outcome of a pairwise comparison where Agent $j$ judges Agent $k$ at time step $t$.
*   $w_{j \to k}^t = 1$ if Agent $j$ ranks $k$ higher than a specific opponent (or higher than the median).
*   $t=0$: Blind Ranking (Round 2).
*   $t=final$: Final Ranking (Round 4).

**2.2. Win Rate (The Score)**
We define the **Score** of Agent $k$ as judged by Agent $j$ as their normalized win fraction:
$$ Sc_{j \to k}^t = \text{WinRate}(j \text{ evaluating } k \text{ at time } t) \in [0, 1] $$

**2.3. Consensus Value (The Truth)**
The "True Value" of Agent $k$ is the average score assigned by all *other* peers.
*   **Pre-Debate Value:** $V_k^{t0} = \frac{1}{N-1} \sum_{j \neq k} Sc_{j \to k}^{t0}$
*   **Post-Debate Value:** $V_k^{final} = \frac{1}{N-1} \sum_{j \neq k} Sc_{j \to k}^{final}$

---

## 3. Reward Engineering

We construct a composite reward function $R_{total}$. We do not use a single scalar reward for the whole response; instead, specific signals reinforce specific behaviors.

### 3.1. The Persuasion Reward ($r_{disc}$) – *The Core Evolution Signal*
**Objective:** Reward agents who successfully identify flaws that the group initially missed. This drives the evolution of Meta-Evaluation.
**Logic:** If Agent $i$ critiques Agent $k$, and the group's valuation of Agent $k$ subsequently **drops** ($V_k^{final} < V_k^{t0}$), Agent $i$ has successfully "persuaded" the group.

Let $\mathcal{T}_i$ be the set of target agents critiqued by Agent $i$ (parsed from `<target>` tags).

$$ r_{disc}(i) = \sum_{k \in \mathcal{T}_i} \text{ReLU}\left( \beta \cdot (V_k^{t0} - V_k^{final}) \right) $$

*   $\beta$: Scaling factor (e.g., $5.0$).
*   **Crucial:** If $V_k^{final} \ge V_k^{t0}$ (the score stayed the same or increased), the reward is **0**. Ineffective critiques are not reinforced.

### 3.2. The Solution Quality Reward ($r_{sol}$)
**Objective:** Reinforce the ability to generate correct, high-quality answers.
**Logic:** Based on the final consensus standing of the agent. We use Z-Score normalization to stabilize PPO training.

$$ r_{sol}(i) = \frac{V_i^{final} - \mu_{batch}}{\sigma_{batch} + \epsilon} $$
*   Where $\mu_{batch}$ and $\sigma_{batch}$ are the mean and std dev of all final values in the current batch.

### 3.3. The Consensus Reward ($r_{meta}$) – *Updated for Small N*
**Objective:** Align the judge's criteria with the "Majority Truth" to prevent random attacks or collusion.
**Logic:** **Majority Vote Alignment**. For every pairwise comparison $(A \text{ vs } B)$, if Agent $i$'s preference matches the group's majority preference, they get a positive reward.

Let $Pref_{group}(A, B) \in \{A, B\}$ be the winner determined by aggregating all agents' votes.
Let $Pref_{i}(A, B) \in \{A, B\}$ be Agent $i$'s vote.

$$ r_{meta}(i) = \frac{1}{|Pairs|} \sum_{(A,B)} \begin{cases} +1 & \text{if } Pref_{i}(A,B) = Pref_{group}(A,B) \\ -1 & \text{if } Pref_{i}(A,B) \neq Pref_{group}(A,B) \end{cases} $$

*   *Note:* Even with small $N$ (e.g., $N=3$), this is robust. If Agent 0 is judging (1 vs 2), the "Group" is the aggregation of Agent 1 and Agent 2's self-evaluations (or external verifier if available, but usually peer consensus).

### 3.4. The Self-Correction Bonus ($r_{accept}$)
**Objective:** Encourage agents to accept valid criticism and improve.
**Logic:** If an agent's revised solution scores higher than their initial solution.

$$ r_{accept}(i) = \mathbb{I}(V_i^{final} > V_i^{t0}) \cdot C $$
*   $C$: A fixed bonus constant (e.g., $0.5$).

---

## 4. RL Implementation Details (Credit Assignment)

We use **Token-Level Masking** in the PPO Loss function. This ensures that the "Persuasion Reward" only updates the "Critique Policy" and not the "Solution Policy."

| Trajectory Segment | Content Description | Assigned Reward Function |
| :--- | :--- | :--- |
| **Segment A** | **Round 1:** Initial Solution Tokens ($S^{init}$) | $w_1 r_{sol} + w_2 r_{accept}$ |
| **Segment B** | **Round 2:** Blind Ranking Tokens | **0 (Masked)** <br> *Reason: Prevents reward hacking on initial bias.* |
| **Segment C** | **Round 2:** Critique Tokens ($C_{i \to k}$) | **$w_3 r_{disc}$** <br> *Reason: Specifically reinforces effective argumentation.* |
| **Segment D** | **Round 3:** Revised Solution Tokens ($S^{rev}$) | $w_1 r_{sol}$ <br> *Reason: Reinforces final answer quality.* |
| **Segment E** | **Round 4:** Final Ranking Tokens | $w_4 r_{meta}$ <br> *Reason: Reinforces accurate judgment.* |

**Hyperparameter Suggestions:**
*   $w_1 (Solution) = 1.0$
*   $w_2 (Accept) = 0.5$
*   $w_3 (Persuasion) = 2.0$ (High weight to drive meta-eval evolution)
*   $w_4 (Meta-Judge) = 1.0$

---

## 5. Summary of the Training Step

1.  **Sample:** Get batch of queries $\mathcal{D}$.
2.  **Rollout:** Run the 3-Round Protocol.
    *   Collect $S^{init}$.
    *   Collect $Sc^{t0}$ (Calculate $V^{t0}$).
    *   Collect $C$ (Parse Targets).
    *   Collect $S^{rev}$.
    *   Collect $Sc^{final}$ (Calculate $V^{final}$).
3.  **Compute Rewards:**
    *   Calculate $r_{disc}$ using the delta between $V^{t0}$ and $V^{final}$.
    *   Calculate $r_{sol}$ and $r_{meta}$ using final standings.
4.  **Update:** Perform PPO update on $\pi_\theta$, masking advantages according to the table in Section 4.
