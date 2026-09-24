"""
Trajectory Harvester
====================
Converts the selected complete EduMCTS trajectory into SFT data.

Important:
- no artificial [Step N] labels;
- teacher interventions remain explicit <reflection type="..."> meta-signals;
- the autonomous rollout continuation is included in the output;
- the harvester NEVER fabricates a missing FINAL ANSWER from the reference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def trajectory_to_cot(
    problem: str,
    trajectory: list[dict],
    reference_answer: str,
    include_teacher_turns: bool = True,
) -> str:
    chunks: list[str] = []

    for turn in trajectory:
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content:
            continue

        if role == "student":
            chunks.append(content)
        elif include_teacher_turns and role.startswith("teacher_"):
            action_type = role.replace("teacher_", "", 1)
            chunks.append(
                f'<reflection type="{action_type}">\n'
                f'{content}\n'
                f'</reflection>'
            )

    return "\n\n".join(chunks).strip()


def trajectory_to_sharegpt(
    problem: str,
    trajectory: list[dict],
    reference_answer: str,
) -> dict:
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"Solve the following problem step by step, showing your reasoning.\n\n{problem}",
            },
            {
                "from": "gpt",
                "value": trajectory_to_cot(
                    problem, trajectory, reference_answer, include_teacher_turns=True
                ),
            },
        ]
    }


def trajectory_to_alpaca(
    problem: str,
    trajectory: list[dict],
    reference_answer: str,
) -> dict:
    return {
        "instruction": "Solve the following problem step by step, showing your reasoning.",
        "input": problem,
        "output": trajectory_to_cot(
            problem, trajectory, reference_answer, include_teacher_turns=True
        ),
    }


class TrajectoryHarvester:
    def __init__(self, min_correct_rate: float = 0.05):
        self.min_correct_rate = min_correct_rate

    def harvest(
        self,
        search_results: list[dict],
        output_path: str,
        format: str = "sharegpt",
    ) -> list[dict]:
        data = []
        skipped = 0

        for result in search_results:
            if result.get("best_reward") != 1.0:
                skipped += 1
                continue

            if result.get("correct_rate", 0.0) < self.min_correct_rate:
                skipped += 1
                continue

            trajectory = result.get("best_trajectory")
            if not trajectory:
                skipped += 1
                continue

            if trajectory[0].get("role") != "student":
                logger.warning("Skip: trajectory does not start with student reasoning.")
                skipped += 1
                continue

            if (trajectory[-1].get("role") != "student" or trajectory[-1].get("truncated")
                    or "FINAL ANSWER:" not in trajectory[-1].get("content", "").upper()):
                logger.warning("Skip: trajectory has no complete final student answer.")
                skipped += 1
                continue

            cot = trajectory_to_cot(
                result["problem"], trajectory, result["reference_answer"]
            )
            if "FINAL ANSWER:" not in cot.upper():
                logger.warning("Skip: selected trajectory lacks FINAL ANSWER.")
                skipped += 1
                continue

            if format == "sharegpt":
                ex = trajectory_to_sharegpt(
                    result["problem"], trajectory, result["reference_answer"]
                )
            elif format == "alpaca":
                ex = trajectory_to_alpaca(
                    result["problem"], trajectory, result["reference_answer"]
                )
            elif format == "cot_only":
                ex = {
                    "problem": result["problem"],
                    "cot": cot,
                    "answer": result["reference_answer"],
                }
            else:
                raise ValueError(f"Unknown format: {format}")

            ex["_meta"] = {
                "correct_rate": result.get("correct_rate"),
                "best_reward": result.get("best_reward"),
                "tree_depth": result.get("tree_depth"),
                "trajectory_length": len(trajectory),
                "partial_turns": sum(bool(t.get("truncated")) for t in trajectory),
                "has_teacher_turns": any(
                    t.get("role", "").startswith("teacher_") for t in trajectory
                ),
                "teacher_turns": result.get("teacher_turns", 0),
                "best_action_sequence": result.get("best_action_sequence"),
                "mean_progress": result.get("mean_progress"),
                "final_selection": result.get("final_selection"),
                "llm_calls": result.get("llm_calls"),
            }
            data.append(ex)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Harvested %d examples; skipped %d.", len(data), skipped)
        return data

    @staticmethod
    def print_stats(training_data: list[dict]) -> None:
        if not training_data:
            print("No harvested examples.")
            return

        rates = [
            x["_meta"]["correct_rate"]
            for x in training_data
            if x.get("_meta", {}).get("correct_rate") is not None
        ]
        progresses = [
            x["_meta"]["mean_progress"]
            for x in training_data
            if x.get("_meta", {}).get("mean_progress") is not None
        ]
        teacher_counts = [x["_meta"].get("teacher_turns", 0) for x in training_data]

        print("=" * 60)
        print(f"Total examples: {len(training_data)}")
        if rates:
            print(f"Average tree correct rate: {sum(rates)/len(rates):.2%}")
        if progresses:
            print(f"Average selected-path progress: {sum(progresses)/len(progresses):.3f}")
        print(f"Average teacher turns: {sum(teacher_counts)/len(teacher_counts):.2f}")
        print("=" * 60)
