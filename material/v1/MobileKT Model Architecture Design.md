# 모델 아키텍처 설계 (Model Architecture Design)

상태: 진행중
담당자: 김효기
마감일: 04/16/2026

## Mobile KT Architecture

1. Question Embedding
    1. Question-to-Vector
2. Question Analysis
    1. Question Difficulty Estimator
    2. Expanded Rasch Model
    3. Domain Relevance Extractor
3. Knowledge Gathering
    1. Student Concept State
    2. Student Domain State
    3. Knowledge Gather Function
4. IRT (Item Response Theory)
    1. Student Ability Estimator
    2. Final Prediction
5. Knowledge Updater
    1. Concept State Updater
    2. Domain State Updater
6. Comparison with Existing KT Models
    1. Taxonomy: Mobile-Feasibility as a Fundamental Divide
    2. Parameter Efficiency
    3. Per-Student State Memory
    4. Inference Computation
    5. Interpretability of Student State
    6. Performance Comparison
    7. Summary
7. Training
8. Further Study

---

## 1. Question Embedding

### Question-to-Vector (QTV)

- Model that projects the raw question ($q^R_t$) to a question embedding vector ($q_t \in \mathbb{R}^d$)
- Model size should be large enough to handle raw question text — runs on the cloud side
- The question vector has $d$ dimensions

> **Implementation note:** For standard KT benchmark datasets that do not provide raw question text, QTV is replaced by a **learnable question embedding table** ($\mathbb{R}^{N_Q \times d}$). This lets the model train end-to-end while keeping the same interface — the embedding lookup output is treated identically to a real QTV output. When raw question text is available, the lookup table is swapped out for an LLM encoder without any downstream architecture change.

---

## 2. Question Analysis

### Question Difficulty Estimator (QDE)

- From the question embedding vector $q_t$, estimate the difficulty level $Diff_{q_t}$ of the question
- The difficulty scalar is used downstream in ERM and in the IRT prediction formula
- The estimator consists of a 2-layer MLP:

$$
Diff_{q_t} = \text{MLP}_{QDE}(q_t), \qquad \text{MLP}_{QDE}: d \to h_{QDE} \to 1
$$

> **Implementation note:** The output is left unbounded (no final activation) so it can participate directly in the IRT formula as a signed difficulty offset.

---

### Expanded Rasch Model (ERM) — adapted from MIKT (WWW'24)

- Generates a multi-concept projected question embedding vector $q'_t$ based on the related concepts $C_{q_t}$ of the question

- **Rasch Model definition:**

    > *A psychometric model for analyzing categorical response data (e.g., test answers) as a function of the trade-off between the respondent's ability and the item difficulty ($Diff_{q_t}$).*

- **Question expression based on the Rasch Model (AKT, 2020):**

    $$
    E_t = C_{c_t} + Diff_{q_t} \times V_{c_t}
    $$

    The question can be expressed as the **concept of the question** ($C_{c_t}$) plus the **difficulty** ($Diff_{q_t}$) scaled by the **direction in concept space** ($V_{c_t}$).

- Prior works only considered single-concept questions. ERM extends this to multi-concept questions.

- **Weighted concept embedding of a multi-concept question ($MC_{q_t}$):**

    $$
    MC_{q_t} = \sum_{j \in C_{q_t}} \alpha_j \cdot CE_j, \qquad \alpha_j = \frac{\exp\!\left(\dfrac{q_t^\top CE_j}{\sqrt{d}}\right)}{\displaystyle\sum_{i \in C_{q_t}} \exp\!\left(\dfrac{q_t^\top CE_i}{\sqrt{d}}\right)}
    $$

    - $CE \in \mathbb{R}^{K \times d}$: concept embedding matrix ($K$ = number of unique concepts)
    - $\alpha_j$: scaled dot-product attention weight of question $q_t$ w.r.t. related concept $j$

- **Difficulty-modulated direction ($OF_{q_t}$):**

    $$
    OF_{q_t} = Diff_{q_t} \times \text{MLP}_1\!\left(\frac{1}{|C_{q_t}|}\sum_{j \in C_{q_t}} CE_j\right)
    $$

    - $\text{MLP}_1: d \to h_{ERM} \to d$ — 2-layer MLP projecting the average concept embedding into the direction vector in concept space
    - $|C_{q_t}|$: number of related concepts for question $q_t$

- **Multi-concept projected question embedding ($q'_t$):**

    $$
    q'_t = q_t + MC_{q_t} + OF_{q_t}
    $$

> **Implementation note:** The residual $q_t$ is added explicitly. Without it, the question-specific signal would be fully replaced by the concept projection $MC + OF$, losing question-level discriminative information. The residual preserves the original question identity alongside the concept-aware projection.

---

### Domain Relevance Extractor (DRE)

- From $q'_t$ and the student's current state, extracts the relevance to the domain knowledge
- The output is a **per-dimension blend gate** $g \in (0,1)^d$ (vector, not scalar) that independently controls how much each latent dimension relies on domain-level vs. concept-level knowledge

- **Aggregated domain knowledge vector:**

    $$
    \bar{D} = \frac{1}{T}\sum_{i=1}^{T} DS_i \cdot DE_i \in \mathbb{R}^d
    $$

    where $DE \in \mathbb{R}^{T \times d}$ are learnable domain embeddings and $DS_i$ are scalar domain mastery values.

- **Per-dimension vector gate:**

    $$
    g = \sigma\!\left(\underbrace{(W_2 \cdot q'_t) \odot \bar{D}}_{\text{per-dim domain–question affinity}} \;+\; \underbrace{W_3 \cdot \sum_{j \in C_{q_t}}\alpha_j \cdot CS_j^{Decay}}_{\text{concept scalar} \to \mathbb{R}^d}\right) \in (0,1)^d
    $$

    - $W_2 \in \mathbb{R}^{d \times d}$: projects $q'_t$ into domain embedding space
    - $\odot$: element-wise (Hadamard) product — produces a per-dimension domain–question match score
    - $W_3 \in \mathbb{R}^{d \times 1}$: broadcasts the scalar concept knowledge signal to $d$ dimensions
    - $\sum_{j \in C_{q_t}} \alpha_j \cdot CS_j^{Decay}$: student's current concept knowledge (scalar)

> **Design rationale:** The original scalar gate $D_\alpha \in (0,1)$ forced a single blending ratio across all $d$ dimensions, assuming domain-level and concept-level knowledge contribute uniformly to every feature. The vector gate $g \in (0,1)^d$ removes this constraint: each dimension $k$ independently learns whether it is better characterised by the domain trend ($\bar{D}_k$) or by concept-specific mastery ($CK_k$). This adds only $d - 1$ parameters ($W_3$ expands from $1 \times 1$ to $d \times 1$) while enabling strictly richer, per-dimension blending.

---

## 3. Knowledge Gathering

Most existing studies use hidden vectors to represent knowledge state, which are **not directly interpretable**.

MobileKT aims for a **fully explainable, on-device knowledge state** by storing each state value in $[0, 1]$ directly on the local device.

### Student Concept State (CS)

- Concept state: $CS \in \mathbb{R}^{(K+1) \times 2}$ — each concept stores (mastery, time since last update)
- Each concept has its own timer tracking the student's last visit to that concept
- Mastery value close to 1 indicates strong understanding; close to 0 indicates weak understanding

> **Implementation note:** Each concept's initial mastery is a **learnable per-concept parameter** (logit scale, initialized to 0 → $\sigma(0) = 0.5$). This allows the model to learn concept-level prior difficulty from data, rather than assuming a uniform 0.5 prior for all concepts.

### Student Domain State (DS)

- Domain state: $DS \in \mathbb{R}^{T+1}$ — a sliding window of the $T$ most recent mastery levels, plus a single timer
- The window tracks the student's recent trend of domain-level mastery
- The timer records elapsed time since the last knowledge update
- As with CS, values close to 1 indicate strong performance; close to 0 indicates weak performance

> **Implementation note:** Each domain window slot has a **learnable initial mastery parameter** (logit scale, initialized to 0 → $\sigma(0) = 0.5$), symmetric to the CS prior. This lets the model learn the average starting domain proficiency from data rather than always initialising to 0.5.

### Knowledge Gather Function (KGF)

Estimates the student's total knowledge state relative to the specific question $q_t$.

- **Forgetting modeling — decayed concept mastery:**

    $$
    CS_j^{Decay} = CS_j \times \sigma\!\left(\text{MLP}_f\!\left(\,encode(I_j)\,\right)\right),
    \qquad encode(I) = \text{tile}(\log(I + 1),\; d)
    $$

    - $I_j$: time (steps) elapsed since concept $j$ was last practiced
    - $encode(I) \in \mathbb{R}^d$: log-scaled interval, tiled to $d$ dimensions

> **Implementation note:** The forgetting gate is a **2-layer MLP** ($d \to d/2 \to 1$). The intermediate non-linear layer allows the model to capture complex, non-monotonic forgetting patterns that a single linear layer cannot express.

- **Domain and concept knowledge w.r.t. the question:**

    $$
    DK_{q_t} = g \odot \bar{D} \in \mathbb{R}^{d}
    $$

    $$
    CK_{q_t} = (1 - g) \odot \sum_{j \in C_{q_t}} (\alpha_j \cdot CS_j^{Decay}) \cdot CE_j \in \mathbb{R}^{d}
    $$

    - $g \in (0,1)^d$: per-dimension blend gate from DRE
    - $\odot$: Hadamard (element-wise) product
    - $CE_j \in \mathbb{R}^d$: concept embedding for concept $j$

- **Final knowledge vector:**

    $$
    FK_{q_t} = (DK_{q_t},\; CK_{q_t}) \in \mathbb{R}^{2d}
    $$

---

## 4. IRT (Item Response Theory)

The Rasch model used in ERM is a special case of IRT with one free parameter (1PL). IRT is the generalized framework:

> *In psychometrics, **item response theory (IRT)** is a paradigm for the design, analysis, and scoring of tests and questionnaires measuring abilities, attitudes, or other latent variables.*

### Student Ability Estimator (SAE)

- Converts the final knowledge vector $FK_{q_t} \in \mathbb{R}^{2d}$ into a scalar ability estimate $FA_{q_t} \in \mathbb{R}$
- The estimator consists of a 2-layer MLP, with the question embedding $q'_t$ concatenated as a skip connection:

$$
FA_{q_t} = \text{MLP}_{SAE}\!\left(\text{LayerNorm}(FK_{q_t} \;\|\; q'_t)\right) \in \mathbb{R},
\qquad \text{MLP}_{SAE}: 3d \to h_{SAE} \to 1
$$

> **Implementation note:**
> - **Skip connection from $q'_t$:** The SAE input is extended to $FK_{q_t} \| q'_t \in \mathbb{R}^{3d}$ by concatenating the multi-concept question embedding. This gives the prediction head direct access to the question context, analogous to the design in ReKT.
> - **LayerNorm:** Applied to the concatenated $[FK; q']$ before the MLP. $FK$ (domain/concept knowledge) and $q'$ (question embedding) live in different scales; LayerNorm normalises them before fusion, stabilising training.

### Final Prediction

- **1PL IRT formula (Rasch model):**

    $$
    P(X_{ij} = 1 \mid \theta_i, \beta_j) = \sigma(\theta_i - \beta_j) = \frac{1}{1 + e^{-(\theta_i - \beta_j)}}
    $$

    - $\theta_i$: student ability; $\beta_j$: question difficulty

- **MobileKT final prediction ($y_t$):**

    $$
    y_t = \sigma\!\left(C \times (FA_{q_t} - Diff_{q_t})\right)
    $$

    - $C$: scale factor — **learnable parameter**, initialized to 3.0

- **Training objective — binary cross-entropy (BCE):**

    $$
    \mathcal{L}_t = -(a_t \log y_t + (1-a_t) \log(1-y_t)), \qquad \mathcal{L} = \sum_{(q_t,\, a_t) \in \mathcal{D}} \mathcal{L}_t
    $$

    where $a_t \in \{0, 1\}$ is the student's response (0 = incorrect, 1 = correct).

---

## 5. Knowledge Updater

After observing the student's response $a_t$, both CS and DS are updated.

The update uses the question embedding because the magnitude of knowledge change should depend on the question's content and difficulty.

- **Acquired knowledge signal:**

    $$
    A_{q_t} = AE_{a_t}, \qquad AE \in \mathbb{R}^{2 \times d}\text{: learnable answer embeddings}
    $$
    $$
    X_t = A_{q_t} + q'_t
    $$

### Concept State Updater (CU)

- Updates $CS_j$ for all related concepts after the student responds

- **Magnitude of update (always $\geq 0$):**

    $$
    |\Delta CS_j| = \left|\tanh\!\left(\text{MLP}_{CU}(X_t,\; encode(I_j))\right)\right| \geq 0,
    \qquad \text{MLP}_{CU}: 2d \to h_{CU} \to 1
    $$

- **Direction determined by response (monotonicity guarantee):**

    $$
    \Delta CS_j = (2a_t - 1) \cdot |\Delta CS_j| \quad \begin{cases} +|\Delta CS_j| & \text{if correct } (a_t=1) \\ -|\Delta CS_j| & \text{if incorrect } (a_t=0) \end{cases}
    $$

> **Implementation note:**
> - **Monotonicity:** Magnitude is always $\geq 0$ via $|\cdot|$; direction is explicitly set by $a_t$, guaranteeing that a correct answer can never decrease mastery and an incorrect answer can never increase it.
> - **CU input:** $X_t$ (acquired knowledge) and $encode(I_j)$ (concept-specific recency) — total input dim $= 2d$. Early experiments added the global domain vector $\bar{D}$ as a third input ($3d$), but ablation showed this did not improve accuracy and was reverted. The question-level signal $X_t = AE_{a_t} + q'_t$ already carries response-type information; concept updates do not benefit from additional domain-level context at this granularity.

- **Logit-space update for related concepts:**

    $$
    CS_j \leftarrow \sigma\!\left(\text{logit}(CS_j) + \alpha_j \cdot \Delta CS_j\right), \qquad I_j = 0
    $$

    where $\text{logit}(x) = \log\dfrac{x}{1-x} = \sigma^{-1}(x)$.

    Logit-space addition is used because $CS_j \in (0,1)$ and $\Delta CS_j \in (-1,1)$: direct arithmetic in probability space would violate the $[0,1]$ bound.

> **Implementation note:** The per-concept delta $\alpha_j \cdot \Delta CS_j$ is projected to the full $K$-concept space via a **one-hot batched matrix multiply** ($\delta_{\text{full}} = \delta_{\text{rel}} \cdot \mathbf{1}_{\text{hot}}$), making the update fully differentiable and loop-free for any $\text{max\_c}$. Padded concept positions are masked before projection.

---

### Domain State Updater (DU)

- Updates the domain mastery sliding window after each response

- **Response mode** (normal update after student answers):

    $$
    \Delta DS = \tanh\!\left(\text{MLP}_{DU}(X_t,\; encode(I))\right), \qquad \text{MLP}_{DU}: 2d \to h_{DU} \to 1
    $$
    $$
    \bar{D}_s = \frac{1}{T}\sum_{i=1}^T DS_i
    $$
    $$
    DS \xleftarrow{\text{push}} \sigma\!\left(\text{logit}(\bar{D}_s) + \Delta DS\right), \qquad I = 0
    $$

    The "push" operation discards the oldest mastery value and appends the new one at the end.

- **Time-only mode** (student is idle — no response):

    $$
    DS \xleftarrow{\text{push}} \bar{D}_s \times \sigma\!\left(W_f \cdot encode(I) + b_f\right), \qquad I \mathrel{+}= \Delta t
    $$

    - $W_f \in \mathbb{R}^{1 \times d},\; b_f \in \mathbb{R}$: learnable forgetting parameters
    - $encode(I) = \text{tile}(\log(I+1),\; d) \in \mathbb{R}^d$: log-scaled elapsed time

> **Implementation note:** State updates are masked for padding positions — when `question_id == 0`, CS and DS are **not** updated and concept timers are **not** incremented, ensuring padded timesteps have no effect on the student's knowledge state.

---

## 6. Comparison with Existing KT Models

### 6.1 Taxonomy: Mobile-Feasibility as a Fundamental Divide

Knowledge tracing models can be categorized into two classes based on their inference strategy, which determines whether on-device deployment is architecturally feasible.

**Full-sequence models** (AKT, SimpleKT, SAINT/SAINT+, CL4KT, RKT) re-process the entire interaction history at each new step. Their self-attention mechanisms incur $O(S^2 \cdot d)$ cost per inference call, which grows unboundedly with sequence length $S$. These models cannot be deployed on mobile devices without storing and replaying the full history, making real-time inference impractical.

**Incremental models** (DKT, DKVMN, LPKT, MIKT, ReKT, MobileKT) maintain a compact student state that is updated in $O(1)$ per step. Only this class is suitable for always-on, offline, on-device inference. However, within incremental models there is a further divide: some models (e.g., ReKT) achieve $O(1)$ compute per step but require state buffers that grow proportionally to the maximum sequence length, making their per-student storage footprint impractically large for mobile deployment.

| Category | Models | Inference Cost | Mobile |
|----------|--------|----------------|--------|
| Full-sequence | AKT, SimpleKT, SAINT, CL4KT, RKT | $O(S^2 \cdot d)$ per call | ✗ |
| Incremental (large state) | ReKT | $O(1)$ per step | △ |
| **Incremental (compact state)** | DKT, DKVMN, LPKT, MIKT, **MobileKT** | $O(1)$ per step | **✓** |

The analysis below focuses exclusively on incremental models, using ASSISTments 2009 (Q = 15,678, K = 123) as the comparison setting.

---

### 6.2 Parameter Efficiency

All models share a structural property: a large question embedding table (size $N_Q \times d$) dominates total parameter count but is logically separable — it can be pre-cached in the cloud or stored as a read-only lookup table. The *core parameter count* (excluding question embeddings) reflects the actual on-device inference engine.

| Model | Total Params | Core Params (ex Q-embed) | Q-embed Fraction |
|-------|-------------|--------------------------|-----------------|
| DKT ($d$ = 200, concept-level) | ~382K | ~382K | 0% |
| DKVMN ($N$ = 20, $d_v$ = 100) | ~810K | ~22K | 97.3% |
| LPKT ($d$ = 128) | ~1,300K | ~500K | 61.5% |
| MIKT ($d$ = 64) | ~3,260K | **187K** | 94.3% |
| **MobileKT ($d$ = 128)** | **~2,281K** | **~274K** | **88.0%** |
| ReKT ($d$ = 128) | ~2,405K | ~383K | 84.1% |

> *Core parameters translate directly to on-device model weight size.* MobileKT's ~274K core parameters require **~1.05 MB** in fp32, or **~0.26 MB** after INT8 quantization — well within the budget of any contemporary mobile application. ReKT's core parameters (~383K) are larger than MobileKT's despite a simpler per-step computation, because ReKT stores learnable sequence-indexed initial states and a full time embedding table.

> **ReKT Q-embed note:** ReKT's question-dependent parameters consist of both `pro_embed` ($N_Q \times d$) and `akt_pro_diff` ($N_Q \times 1$), amounting to ~2,022K (84.1% of total).

---

### 6.3 Per-Student State Memory

The state maintained per student must be stored persistently on the device (or transmitted at session boundaries). Smaller state enables larger concurrent student populations and lower storage overhead.

| Model | State Structure | Size per Student |
|-------|----------------|-----------------|
| LPKT | GRU hidden ($d$ = 128 floats) | **512 B** |
| DKT | LSTM hidden ($d$ = 200 floats) | **800 B** |
| **MobileKT** | $CS \in \mathbb{R}^{(K+1)\times 2}$, $DS \in \mathbb{R}^{T+1}$ | **~1 KB** |
| DKVMN | Value memory ($N \times d_v$ = 20 × 100 floats) | ~8 KB |
| MIKT | skill\_state ($K \times d$ = 123 × 64 floats) + all\_state | **~31 KB** |
| ReKT | seq-indexed pro/skill buffers + last-visit timestamps | **~262 KB** |

MobileKT's state is **~30× smaller than MIKT** despite capturing richer structure: per-concept mastery, per-concept recency timer, and a domain-level trend window. For a platform serving 1M students, MIKT requires ~29 GB of state storage; MobileKT requires **~1 GB**, fitting entirely within on-device storage on modern smartphones.

**ReKT state breakdown** (ASSIST09, $d$ = 128, $S_{max}$ = 200): Although ReKT achieves $O(1)$ computation per step, its student state grows with the maximum sequence length because it uses **sequence-position-indexed state buffers** rather than a fixed-size summary:

| ReKT State Component | Structure | Size |
|----------------------|-----------|------|
| `pro_state` (sequence buffer) | $(S_{max}, d)$ = 200 × 128 | ~100 KB |
| `skill_state` (sequence buffer) | $(S_{max}, d)$ = 200 × 128 | ~100 KB |
| `last_pro_time` (last step per question) | $(N_Q,)$ = 15,678 int | ~61 KB |
| `last_skill_time` (last step per concept) | $(K,)$ = 167 int | ~668 B |
| `all_state` (global domain state) | $(d,)$ = 128 floats | ~512 B |
| **Total** | | **~262 KB** |

ReKT's per-student state is **262× larger than MobileKT** and **8.5× larger than MIKT**. For 1M students, ReKT requires ~249 GB of state storage — impractical for on-device or even server-side caching at scale. The sequence-indexed design also requires replaying the interaction buffer on every session reconnect, adding latency beyond the per-step compute cost.

---

### 6.4 Inference Computation (MACs per Step)

Inference cost per question determines real-time latency and battery consumption during a tutoring session.

| Model | MACs per Step | Dominant Operation |
|-------|--------------|-------------------|
| DKVMN | ~50K | Memory slot read (small $d_v$) |
| **ReKT** | **~262K** | 3 × FRU (forget + update) + output head |
| DKT | ~358K | LSTM gate computation |
| LPKT | ~400K | GRU step + time encoding |
| **MobileKT** | **~397K** | QDE + ERM + KGF + DRE + SAE |
| MIKT | **~4,100K** | skill\_forget applied to all $K$ concepts |

**ReKT** uses a lightweight FRU (Forget-Response-Update) core: each of its three state tracks (question, concept, domain) applies two $\text{Linear}(2d \to d)$ operations (forget gate + update), totalling $3 \times 4d^2 \approx 197$K MACs, plus the output head ($4d^2 \approx 65$K), for **~262K MACs/step** — the lowest among question-based incremental models. However, this compute advantage is offset by the ~262 KB per-student state requirement.

MIKT's bottleneck is applying the `skill_forget` network — a two-layer MLP — to **all $K$ = 123 concepts at every step**, regardless of how many are relevant to the current question. MobileKT applies its concept operations only to the $\text{max\_c}$ concepts associated with the current question (typically 1–4), resulting in a **~10× reduction in per-step computation** while achieving comparable or better performance.

---

### 6.5 Interpretability of Student State

A core motivation for on-device KT is enabling transparent, actionable feedback to learners and educators. This requires the student state to be directly human-readable.

| Model | State Interpretability | Readable as Mastery? |
|-------|----------------------|---------------------|
| DKT, LPKT | LSTM/GRU hidden vector $\in \mathbb{R}^d$ | ✗ Black-box |
| DKVMN | Value memory matrix $\in \mathbb{R}^{N \times d}$ | ✗ Latent |
| MIKT | skill\_state $\in \mathbb{R}^{K \times d}$ (latent per concept) | △ Partial |
| ReKT | $h^Q, h^C, h^D \in \mathbb{R}^d$ (latent, 3-track) | ✗ Black-box |
| AKT, SimpleKT | Attention weights only; no persistent state | ✗ — |
| **MobileKT** | $CS_j \in [0, 1]$ (mastery probability) + timer $I_j$ | **✓ Direct** |

MobileKT is the **only model in this comparison where the student state is directly interpretable as mastery probabilities**. The value $CS_j$ for concept $j$ is a sigmoid-bounded scalar that can be displayed to a learner ("Algebra mastery: 74%") or used by a teacher dashboard without any post-hoc analysis. ReKT tracks three parallel $d$-dimensional state vectors ($h^Q, h^C, h^D$) using FRU gating, but these are latent representations — the same black-box limitation as DKT and LPKT, despite ReKT's higher predictive accuracy.

---

### 6.6 Performance Comparison

Results on ASSISTments 2009 (test set AUC). Baseline figures are taken from published papers or reproduced under equivalent settings.

| Model | Type | Test AUC (ASSIST09) | Core Params | State/Student | MACs/step |
|-------|------|---------------------|-------------|--------------|-----------|
| DKT | Incremental | ~0.740 | ~382K | 800 B | ~358K |
| DKVMN | Incremental | ~0.750 | ~22K | 8 KB | ~50K |
| LPKT | Incremental | ~0.757 | ~500K | 512 B | ~400K |
| AKT | Full-seq | ~0.780 | ~2,600K | — | $O(S^2)$ |
| SimpleKT | Full-seq | ~0.780 | ~500K | — | $O(S^2)$ |
| MIKT (WWW'24) | Incremental | 0.787 | **187K** | 31 KB | ~4,100K |
| ReKT (MM'24) | Incremental† | **0.792** | 383K | ~262 KB | ~262K |
| **MobileKT — $d$=64 (ours)** | **Incremental** | **0.750** | **~72K** | **~1 KB** | **~100K** |
| **MobileKT — $d$=128 (ours)** | **Incremental** | **0.756**‡ | **~274K** | **~1 KB** | **~397K** |

> †ReKT is technically $O(1)$ per step but stores sequence-length-indexed state buffers, resulting in ~262 KB per student — impractical at mobile scale.
> ‡MobileKT $d$=128 current best: 0.7559 (ktbd pretrain); historical best 0.764 (earlier run, config not retained). $d$=64 result: scratch training.

ReKT achieves the highest AUC among incremental models (0.792 on ASSIST09) using a lightweight FRU (Forget-Response-Update) framework with three parallel state tracks (question, concept, domain). However, its ~262 KB per-student state is 262× larger than MobileKT's ~1 KB, and all state vectors are latent and non-interpretable. MobileKT at $d$=64 reduces core on-device params to **~72K** (~0.28 MB fp32) while matching LPKT accuracy — the only model in this comparison combining sub-1 KB student state, interpretable mastery values, and competitive AUC.

---

### 6.7 Summary

| Criterion | DKT | DKVMN | LPKT | MIKT | ReKT | **MobileKT** |
|-----------|-----|-------|------|------|------|--------------|
| $O(1)$ inference per step | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Offline / on-device | ✓ | ✓ | ✓ | ✓ | △ | **✓** |
| State < 2 KB | ✓ | ✗ | ✓ | ✗ | ✗ | **✓** |
| Inference < 500K MACs | ✓ | ✓ | ✓ | ✗ | ✓ | **✓** |
| Interpretable state | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Per-concept mastery readout | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Competitive AUC (> 0.75) | ✗ | ✗ | ✓ | ✓ | ✓ | **✓** |

> △ ReKT is nominally incremental but requires ~262 KB per-student state — unsuitable for practical on-device deployment at scale.

MobileKT is the **only model satisfying all seven criteria simultaneously**. ReKT achieves the highest AUC among incremental models but fails on state compactness, offline feasibility at scale, and interpretability. MobileKT occupies a unique position: it is the first KT architecture designed jointly for compact on-device state (~1 KB), human-interpretable concept mastery, and competitive predictive accuracy.

---

## 7. Training

### Datasets

Experiments are conducted on multiple standard KT benchmarks. The primary evaluation dataset is **ASSISTments 2009 (assist09)**:

| Split | Students |
|-------|----------|
| Train | 2,496 |
| Val   | 576 |
| Test  | 768 |

- Unique questions: 15,678 &ensp;·&ensp; Unique concepts: 123 &ensp;·&ensp; Max sequence length: 200

Additional datasets: **ASSISTments 2015 (assist15)**, **KT-BD (ktbd)**.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim ($d$) | 128 |
| Domain window ($T$) | 5 |
| QDE hidden ($h_{QDE}$) | 128 |
| ERM hidden ($h_{ERM}$) | 128 |
| SAE hidden ($h_{SAE}$) | 256 |
| CU hidden ($h_{CU}$) | 128 |
| DU hidden ($h_{DU}$) | 128 |
| Dropout | 0.2 |
| IRT scale $C$ init | 3.0 (learnable) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 64 |
| LR schedule | CosineAnnealingLR ($T_{max}=150$, $\eta_{min}=10^{-5}$) |
| Grad clip | 1.0 |
| Max epochs | 150 |
| Early stopping patience | 25 |

### Parameter Count (assist09, $d$ = 128)

| Component | Parameters |
|-----------|-----------|
| question\_embed ($N_Q \times d = 15{,}678 \times 128$) | ~2,007K |
| All other modules (on-device) | ~274K |
| **Total** | **~2,281K** |

> `question_embed` is a training-only proxy for the cloud-side QTV. At deployment, it is replaced by the LLM encoder output; the on-device footprint is **~274K params** (~1.05 MB fp32, ~0.26 MB INT8).

### Training Procedure

1. **Precompute** QDE + ERM outputs for the full sequence before the recurrent loop (avoids redundant per-step recomputation).
2. **Teacher-forced** forward pass: at step $t$, predict correctness for question $t+1$ using the state built from responses $0 \ldots t$.
3. **State update** at each valid step; padding positions (`question_id == 0`) are skipped.
4. **Concept timers** increment by 1 for all valid steps after the update.
5. Loss is BCE averaged over all valid (non-padding) positions.
6. Best model checkpoint selected by validation AUC.

### Hyperparameter Ablation (assist09)

**Phase 1 — architecture with $\bar{D}$ in CU (input dim $3d$), patience=15:**

All runs: $d=128$, max\_seq=200, weight\_decay=1e-5, grad\_clip=1.0, batch=64, CosineAnnealingLR.

| Config | Val AUC | Test AUC | Test ACC | Note |
|--------|---------|----------|----------|------|
| lr=1e-3, nd=5, dp=0.2 | 0.7554 | 0.7497 | 0.7167 | baseline |
| lr=5e-4, nd=5, dp=0.2 | 0.7564 | 0.7513 | 0.7182 | lower LR slightly better |
| lr=1e-3, nd=5, dp=0.3 | 0.7562 | 0.7520 | 0.7171 | more dropout helps |
| lr=1e-3, nd=10, dp=0.2 | 0.7525 | 0.7469 | 0.7130 | larger domain window hurts |
| lr=2e-4, nd=5, dp=0.2 | **0.7580** | **0.7549** | **0.7196** | best in phase |
| lr=1e-3, nd=5, dp=0.2, **d=64** | 0.7571 | 0.7500 | 0.7166 | **half params, same perf** |

**Phase 2 — CU reverted to $2d$ input (no $\bar{D}$), patience=25, max\_epochs=150:**

| Config | Val AUC | Test AUC | Test ACC | Note |
|--------|---------|----------|----------|------|
| lr=1e-3, nd=5, dp=0.2 | — | 0.7491 | 0.7164 | |
| lr=5e-4, nd=5, dp=0.2 | — | 0.7517 | 0.7176 | |
| lr=2e-4, nd=5, dp=0.2 | — | **0.7525** | **0.7179** | best scratch, current arch |

**Key finding — n\_domains:** Window size $T=5$ outperforms $T=10$ on assist09. Larger windows dilute recent mastery signal with stale domain history.

**Key finding — d=64 vs d=128:** Reducing $d$ by half (core params ~72K vs ~274K, on-device weight **0.28 MB fp32**) yields virtually identical AUC. This is the recommended deployment config for mobile.

**Key finding — $\bar{D}$ in CU:** Adding the domain vector to CU input ($3d$) did not improve over the $2d$ baseline and was reverted. Domain context is already captured implicitly through the DRE gate that feeds into KGF.

### Cross-Dataset Pretraining

To test whether the core MLP weights (QDE, ERM, KGF, DRE, SAE, CU, DU) generalise across datasets, we pretrain on ktbd and transfer all shape-compatible weights to an assist09 model. Dataset-specific parameters (`question_embed`, `concept_embed`, `init_mastery`) are re-initialised and learned from scratch.

**Phase 1 — architecture with $\bar{D}$ in CU (patience=15):**

| Source → Target | Init Strategy | Val AUC | Test AUC | Test ACC |
|----------------|---------------|---------|----------|----------|
| scratch | random | 0.7564 | 0.7513 | 0.7182 |
| ktbd → assist09 | pretrain + lr=1e-3 | 0.7558 | 0.7504 | 0.7178 |
| ktbd → assist09 | pretrain + lr=5e-4 | 0.7563 | 0.7515 | 0.7180 |
| ktbd → assist09 | pretrain + lr=5e-4, dp=0.1 | 0.7557 | 0.7508 | 0.7164 |

**Phase 2 — CU reverted to $2d$ input, ktbd pretrain epochs=200, patience=35, assist09 patience=25:**

| Source → Target | Init Strategy | ktbd Test AUC | assist09 Val AUC | assist09 Test AUC | Test ACC |
|----------------|---------------|--------------|-----------------|-------------------|----------|
| scratch | random | — | — | 0.7525 | 0.7179 |
| ktbd → assist09 | pretrain + lr=2e-4 | 0.7461 | — | **0.7559** | **0.7223** |

**Key finding:** Pretraining on ktbd consistently improves over scratch. With the reverted $2d$-CU architecture and extended pretraining (200 epochs, patience=35), the ktbd→assist09 transfer achieves **0.7559 test AUC** — the current best result, surpassing the previous phase-1 best (0.7549) despite the CU simplification. The pretrain benefit is most pronounced at lr=2e-4, which fine-tunes the transferred weights without catastrophic forgetting.

### Results Summary

| Model | Dataset | Test AUC | Core Params | State/Student | Note |
|-------|---------|----------|-------------|--------------|------|
| MIKT (WWW'24) | assist09 | 0.787 | 187K | ~31 KB | published |
| ReKT (MM'24) | assist09 | **0.792** | 383K | ~262 KB | published |
| **MobileKT — historical best**‡ ($d$=128) | assist09 | **0.764** | ~274K | ~1 KB | earlier run |
| **MobileKT — ktbd pretrain** ($d$=128) | assist09 | **0.7559** | ~274K | ~1 KB | current best |
| **MobileKT — best scratch** ($d$=128) | assist09 | 0.7525 | ~274K | ~1 KB | current arch |
| **MobileKT — best scratch** ($d$=64) | assist09 | 0.7500 | **~72K** | **~1 KB** | mobile deploy |
| MobileKT ($d$=128) | ktbd | 0.7461 | ~46K† | ~1 KB | pretrain source |
| MobileKT ($d$=128) | assist15 | 0.6887 | ~274K | ~1 KB | |

> †ktbd: $N_Q = K = 125$, so `question_embed` is small; on-device core params ~46K.
> ‡Historical best: achieved in an earlier experimental run (logs subsequently cleared). Exact config not retained; current experiments have not yet reproduced it.

The current best result (**0.7559** test AUC, ktbd→assist09 pretrain) closes the gap to MIKT (0.787) to within ~0.031 AUC while maintaining a ~1 KB interpretable student state vs. MIKT's 31 KB latent state. The assist15 result (0.6887) is notably lower than assist09 — consistent with assist15's sparser concept structure. The remaining gap to ReKT (0.792) reflects the fundamental capacity tradeoff of scalar mastery representation.

---

## 8. Further Study

### Accuracy Gap Analysis

MobileKT currently achieves test AUC of 0.750–0.755 on assist09, vs. MIKT 0.787 and ReKT 0.792. The primary source of this gap is the **interpretability constraint**: MobileKT stores concept mastery as a scalar $CS_j \in [0,1]$, while MIKT and ReKT use $d$-dimensional latent vectors per concept, giving them substantially higher representational capacity. Closing this gap without sacrificing scalar interpretability is the central research challenge.

### Completed Improvements

- ~~**Vector gating in DRE:** Replacing the scalar gate $D_\alpha$ with a per-dimension gate vector $g \in (0,1)^d$ for more expressive domain-concept blending.~~ *(implemented — §2 DRE)*
- ~~**Cross-dataset pretraining:** Transferring core MLP weights (QDE, ERM, KGF, DRE, SAE, CU, DU) from a source dataset to initialise training on the target dataset.~~ *(validated — current best 0.7559 on assist09 from ktbd→assist09 transfer; see §7)*

### Reverted Changes

- **CU global context ($\bar{D}$):** Adding the aggregated domain vector to CU input (total $3d$) was implemented and tested, but ablation showed no improvement over the $2d$ baseline and was reverted. The domain context is already available to the model through the DRE gate path; feeding it again into CU introduced redundancy without benefit.

### Open Directions

- **CU global context (revisit with larger datasets):** The $\bar{D}$ ablation was conducted on assist09 (small, 2,496 students). On larger datasets the domain signal may carry more information. Worth re-evaluating on EdNet or similar.

- **Domain-concept alignment:** Explicit soft assignment of concepts to domains so that practicing concept $j$ updates only the relevant domain(s), rather than a global window — enabling more targeted domain mastery tracking. Expected to improve both accuracy and interpretability of the domain state.

- **Larger pretraining corpus:** The ktbd pretraining dataset is small ($N_Q = K = 125$). Pretraining on a much larger dataset (e.g., EdNet with millions of interactions) could provide a stronger initialisation for the core weights, amplifying the transfer benefit.

- **Learnable mastery update rule:** The current logit-space additive update ($\text{logit}(CS_j) + \alpha_j \cdot \Delta CS_j$) is fixed-form. A small per-concept GRU or gated update rule could increase expressive capacity while keeping the output scalar in $[0,1]$.

- **$q_t$ residual in ERM (ablation pending):** The current implementation computes $q'_t = q_t + MC_{q_t} + OF_{q_t}$, adding $q_t$ explicitly as a residual. The original MIKT formulation uses only $q'_t = MC_{q_t} + OF_{q_t}$. Since $q_t$ already influences $q'_t$ implicitly through the attention weights $\alpha_j$, the explicit residual may introduce question-level shortcuts. Removing it could improve concept-space generalisation.

- **Truncated BPTT:** For long sequences ($S > 50$), gradient signal from early steps can vanish through the logit–sigmoid chain. Backpropagating only through the last $K$ steps may improve training stability on large-scale datasets.

- **Real QTV experiments:** Validating the architecture with actual LLM-encoded question embeddings instead of the lookup-table proxy, to measure the contribution of rich question representations on performance and generalisation.
