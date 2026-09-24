"""
EduMCTS Configuration
=====================
Clean pedagogical MCTS with transition-progress shaping.

Design:
1. The root contains one autonomous student reasoning step.
2. Every later state can expand four equal actions:
      {self, hint, critique, socratic}
3. Each expansion produces exactly one new student reasoning step.
4. A lightweight evaluator scores only the realized before -> after progress
   of that transition on {-2,-1,0,+1,+2}.
5. Standard UCT operates on one unified return:
      G = terminal_correctness + progress_weight * mean_path_progress
6. The final pedagogical path is chosen by tree visit statistics, never by
   teacher count, DCS, or a post-hoc teacher reranker.
"""

from dataclasses import dataclass, field
import os
from urllib.parse import urlparse


@dataclass
class ModelConfig:
    # Supply your own OpenAI-compatible service through environment variables.
    base_url: str = field(default_factory=lambda: os.getenv("EDUMCTS_BASE_URL", "<YOUR_API_BASE_URL>"), repr=False)
    api_key: str = field(default_factory=lambda: os.getenv("EDUMCTS_API_KEY", "<YOUR_API_KEY>"), repr=False)
    model_name: str = field(default_factory=lambda: os.getenv("EDUMCTS_MODEL", "qwen3-235b-a22b-instruct-2507"))

    timeout: int = 180
    max_retries: int = 3

    teacher_temperature: float = 0.35
    student_step_temperature: float = 0.75
    rollout_temperature: float = 0.70
    evaluator_temperature: float = 0.0
    verifier_temperature: float = 0.0

    teacher_max_tokens: int = 256
    student_step_max_tokens: int = 384
    rollout_max_tokens: int = 2048
    evaluator_max_tokens: int = 256
    verifier_max_tokens: int = 256
    generation_repair_attempts: int = 2
    verifier_format_retries: int = 2

    def validate(self) -> None:
        """Reject example credentials before making any API requests."""
        missing = []
        if not self.api_key.strip() or self.api_key.strip() in {
            "<YOUR_API_KEY>", "YOUR_API_KEY"
        }:
            missing.append("EDUMCTS_API_KEY")
        endpoint = urlparse(self.base_url.strip())
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.netloc
            or "<" in self.base_url
            or "YOUR_API_BASE_URL" in self.base_url
        ):
            missing.append("EDUMCTS_BASE_URL")
        if not self.model_name.strip():
            missing.append("EDUMCTS_MODEL")
        if missing:
            raise ValueError(
                "Set " + ", ".join(missing)
                + " before running synthesis; see .env.example."
            )


@dataclass
class MCTSConfig:
    # Depth counts STUDENT tree steps. The forced initial Self step is depth 1.
    max_depth: int = 10
    num_rollouts: int = 20

    # Standard UCT.
    exploration_c: float = 1.4

    # Dense pedagogical shaping. Evaluator raw score is in [-2, 2] and is
    # normalized to [-1, 1] before aggregation. Keeping this < 1 ensures
    # terminal correctness remains the dominant signal.
    progress_weight: float = 0.25

    actions: list = field(
        default_factory=lambda: ["self", "hint", "critique", "socratic"]
    )

    correct_reward: float = 1.0
    wrong_reward: float = 0.0

    # Harvest only trees that produced at least this fraction of correct search
    # rollouts. Final trajectory correctness is checked separately.
    min_correct_rate: float = 0.05


@dataclass
class DataConfig:
    seed_data_path: str = "data/seed_problems.json"
    output_dir: str = "output"


MODEL_CFG = ModelConfig()
MCTS_CFG = MCTSConfig()
DATA_CFG = DataConfig()
