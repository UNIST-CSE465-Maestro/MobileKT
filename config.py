from dataclasses import dataclass


@dataclass
class MobileKTConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "assist2009"       # assist2009 | assist2012 | assist2015 | statics
    data_dir: str = "data"
    max_seq_len: int = 200

    # ── Vocabulary sizes (set automatically during data loading) ─────────────
    n_questions: int = 17751          # total number of unique questions
    n_concepts: int = 123             # K: total number of unique concepts
    n_domains: int = 10               # T: domain window size (recent T masteries kept)

    # ── Model dimensions ─────────────────────────────────────────────────────
    d: int = 64                       # embedding dimension (all vectors share this)

    # ── QDE / ERM ────────────────────────────────────────────────────────────
    # QDE: 3-layer MLP  d → qde_hidden → qde_hidden//2 → 1
    qde_hidden: int = 128             # hidden size inside QDE MLP
    erm_hidden: int = 128             # hidden size inside ERM direction transform
    qe_input_mode: str = "features"   # v4: "features" raw-text cache | "id" MIKT-ID baseline
    question_feature_dim: int | None = None  # v4: cached raw-text encoder feature dim
    question_features_path: str = ""   # v4: path to question feature matrix (.pt/.npy)

    # ── IRT ──────────────────────────────────────────────────────────────────
    irt_scale: float = 3.0            # C scalar: smooths sigmoid output
    # SAE: 3-layer MLP  3d → sae_hidden → sae_hidden//2 → 1
    sae_hidden: int = 256             # hidden size inside SAE MLP

    # ── Updater ──────────────────────────────────────────────────────────────
    cu_hidden: int = 128              # Concept Updater hidden size
    du_hidden: int = 128              # Domain Updater hidden size
    use_diff_bias: bool = True        # v3 only: per-question difficulty bias scalar
    mikt_state_dim: int = 64          # v4 only: MIKT per-concept state dimension
    mikt_output_scale: float = 5.0    # v4 only: Rasch-style ability-difficulty scale

    # ── Training ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    n_epochs: int = 100
    dropout: float = 0.2
    patience: int = 10                # early stopping patience
    grad_clip: float = 1.0

    # ── Experiment ───────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"              # cuda | cpu
    save_dir: str = "experiments"
    wandb: bool = False
    session: str = ""                 # shared timestamp folder for concurrent runs
    pretrain_ckpt: str = ""           # path to a pretrained checkpoint for core weight transfer

    # ── QTV (cloud-side) ─────────────────────────────────────────────────────
    # For standard KT benchmarks (no raw text), QTV acts as a simple
    # learnable embedding lookup. Replace with a real LLM encoder for
    # raw-question experiments.
    qtv_mode: str = "embed"           # "embed" (lookup) | "llm" (cloud LLM encoder)
