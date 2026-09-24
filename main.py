"""
EduMCTS Main
============
Runs clean pedagogical MCTS on an existing JSON problem set.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from config import MODEL_CFG, MCTS_CFG, DATA_CFG
from llm_client import LLMClient
from mcts import EduMCTS
from harvester import TrajectoryHarvester

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("edumcts.main")


def project_path(path: str) -> Path:
    """Resolve relative input/output paths against this project, not the shell."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )


def demo_problems() -> list[dict]:
    return [
        {
            "problem": "A bag contains 3 red and 5 blue balls. Two balls are drawn without replacement. What is the probability that both are red?",
            "answer": "3/28",
        },
        {
            "problem": "Prove that for every integer n, n^2+n is even.",
            "answer": "n^2+n=n(n+1); one of two consecutive integers is even.",
        },
    ]


def load_problems(path: str) -> list[dict]:
    p = project_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Problem file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of problem objects.")

    out = []
    for i, item in enumerate(raw):
        problem = item.get("problem") or item.get("input")
        answer = item.get("answer") or item.get("reference_answer")
        if problem and answer is not None:
            out.append({
                "id": item.get("id", i),
                "problem": str(problem),
                "answer": str(answer),
            })
    return out


def slice_problems(problems: list[dict], spec: list[int]) -> list[dict]:
    if len(spec) == 1:
        return problems[:spec[0]]
    if len(spec) == 2:
        start = max(0, spec[0] - 1)
        end = spec[1]
        return problems[start:end]
    raise ValueError("--problems accepts either N or START END")


def run_one(prob: dict, args) -> dict:
    model_cfg = copy.deepcopy(MODEL_CFG)
    mcts_cfg = copy.deepcopy(MCTS_CFG)
    mcts_cfg.num_rollouts = args.rollouts
    mcts_cfg.max_depth = args.depth
    mcts_cfg.progress_weight = args.progress_weight
    mcts_cfg.exploration_c = args.exploration_c

    llm = LLMClient(model_cfg)
    searcher = EduMCTS(llm, mcts_cfg)
    result = searcher.search(prob["problem"], prob["answer"])
    result["problem_id"] = prob.get("id")
    result["verifier_diagnostics"] = llm.verifier_diagnostics
    return result


def run_batch(problems: list[dict], args) -> list[dict]:
    results: list[dict] = [None] * len(problems)  # type: ignore

    def task(idx, prob):
        try:
            return idx, run_one(prob, args)
        except Exception as e:
            logger.exception("Problem %s failed: %s", prob.get("id", idx), e)
            return idx, {
                "problem_id": prob.get("id"),
                "problem": prob["problem"],
                "reference_answer": prob["answer"],
                "best_trajectory": None,
                "best_reward": 0.0,
                "correct_rate": 0.0,
                "error": str(e),
                "verifier_diagnostics": getattr(e, "diagnostics", []),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, i, p) for i, p in enumerate(problems)]
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
            logger.info(
                "[%d/%d] reward=%s correct_rate=%.1f%% mean_progress=%s selection=%s",
                idx + 1,
                len(problems),
                result.get("best_reward"),
                100.0 * result.get("correct_rate", 0.0),
                result.get("mean_progress"),
                result.get("final_selection"),
            )
    return results


def save_outputs(results: list[dict], output_dir: str) -> None:
    out = project_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "raw_search_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    harvester = TrajectoryHarvester(min_correct_rate=MCTS_CFG.min_correct_rate)
    sharegpt = harvester.harvest(
        results, str(out / "training_data_sharegpt.json"), format="sharegpt"
    )
    harvester.harvest(
        results, str(out / "training_data_alpaca.json"), format="alpaca"
    )
    harvester.harvest(
        results, str(out / "training_data_cot.json"), format="cot_only"
    )
    harvester.print_stats(sharegpt)


def main():
    parser = argparse.ArgumentParser(description="Clean pedagogical EduMCTS")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--input", default=DATA_CFG.seed_data_path)
    parser.add_argument("--problems", type=int, nargs="+", default=[2])
    parser.add_argument("--rollouts", type=int, default=MCTS_CFG.num_rollouts)
    parser.add_argument("--depth", type=int, default=MCTS_CFG.max_depth)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-weight", type=float, default=MCTS_CFG.progress_weight)
    parser.add_argument("--exploration-c", type=float, default=MCTS_CFG.exploration_c)
    parser.add_argument("--output-dir", default=DATA_CFG.output_dir)
    args = parser.parse_args()

    try:
        MODEL_CFG.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.workers < 1 or args.rollouts < 1 or args.depth < 1:
        parser.error("--workers, --rollouts and --depth must be positive")
    if len(args.problems) not in {1, 2} or any(n < 1 for n in args.problems):
        parser.error("--problems expects a positive count or a 1-based START END range")
    if len(args.problems) == 2 and args.problems[0] > args.problems[1]:
        parser.error("--problems START must not exceed END")

    if args.mode == "demo":
        problems = slice_problems(demo_problems(), args.problems)
    else:
        problems = slice_problems(load_problems(args.input), args.problems)

    if not problems:
        parser.error("No problems selected from the input data")
    setup_logging(project_path(args.output_dir))

    logger.info(
        "Model=%s | problems=%d | rollouts=%d | depth=%d | progress_weight=%.3f | c=%.3f",
        MODEL_CFG.model_name,
        len(problems),
        args.rollouts,
        args.depth,
        args.progress_weight,
        args.exploration_c,
    )
    results = run_batch(problems, args)
    save_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
