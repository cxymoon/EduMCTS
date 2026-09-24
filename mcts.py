"""
EduMCTS Core — clean pedagogical MCTS
=====================================

Root:
    problem + one autonomous student reasoning step

Tree edge:
    choose one of {Self, Hint, Critique, Socratic}
    -> optional teacher intervention
    -> exactly one new student reasoning step
    -> one transition-progress score in {-2,-1,0,+1,+2}

Simulation:
    one autonomous student continuation from the selected tree node to FINAL ANSWER

Search return:
    terminal correctness + progress_weight * mean transition progress on the
    selected tree path. Progress scores are normalized to [-1,1].

Selection:
    standard UCT only. No DCS/PA-UCT/process-value side channel.

Final pedagogical path:
    principal variation obtained by repeatedly choosing the most visited child.
    No teacher-count bonus and no post-hoc teacher/LLM reranker.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
import math
import random
from typing import Optional

from config import MCTSConfig, MCTS_CFG
from llm_client import LLMClient

logger = logging.getLogger(__name__)


def _has_final_answer(text: str) -> bool:
    return "FINAL ANSWER:" in (text or "").upper()


@dataclass
class MCTSNode:
    conversation_history: list[dict] = field(default_factory=list)
    step_count: int = 0

    visit_count: int = 0
    total_return: float = 0.0

    parent: Optional["MCTSNode"] = None
    action_from_parent: Optional[str] = None
    children: dict[str, "MCTSNode"] = field(default_factory=dict)

    # Evaluator result for the edge parent -> this node. Root has no such edge.
    transition_eval: Optional[dict] = None
    transition_progress: float = 0.0  # normalized to [-1, 1]

    is_terminal: bool = False
    is_correct: Optional[bool] = None
    terminal_verifier: Optional[dict] = None

    # Stored only to avoid wasting a second rollout when the principal leaf has
    # already produced a correct continuation during search.
    successful_trajectory: Optional[list[dict]] = None
    successful_verifier: Optional[dict] = None

    @property
    def q_value(self) -> float:
        return self.total_return / self.visit_count if self.visit_count else 0.0

    def uct(self, parent_visits: int, c: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploration = c * math.sqrt(
            math.log(max(1, parent_visits) + 1.0) / self.visit_count
        )
        return self.q_value + exploration

    def is_fully_expanded(self, actions: list[str]) -> bool:
        return all(a in self.children for a in actions)

    def best_child(self, c: float) -> "MCTSNode":
        return max(
            self.children.values(),
            key=lambda ch: ch.uct(self.visit_count, c),
        )

    def add_child(self, action: str, child: "MCTSNode") -> "MCTSNode":
        child.parent = self
        child.action_from_parent = action
        self.children[action] = child
        return child


class EduMCTS:
    def __init__(self, llm: LLMClient, cfg: MCTSConfig = MCTS_CFG):
        self.llm = llm
        self.cfg = cfg

    def search(
        self,
        problem: str,
        reference_answer: str,
        rollout_callback=None,
    ) -> dict:
        """
        rollout_callback(rollout_idx, terminal_reward, node, correct_count, *,
        mean_progress, search_return) is invoked after each search rollout. It
        exists so experiments can recover the nested budget curve -- best-so-far
        after the first k rollouts -- from a single search, without paying for a
        fresh run per budget. When omitted the search behaves exactly as before.
        """
        root, root_info = self._initialize_root(problem, reference_answer)

        if root_info["initial_self_correct"]:
            return {
                "problem": problem,
                "reference_answer": reference_answer,
                "best_trajectory": copy.deepcopy(root.conversation_history),
                "best_reward": 1.0,
                "correct_rate": 1.0,
                "total_rollouts": 0,
                "tree_depth": 0,
                "initial_self_step": root_info["initial_self_step"],
                "initial_self_correct": True,
                "initial_self_wrong_final": False,
                "best_action_sequence": ["initial_self"],
                "teacher_turns": 0,
                "mean_progress": 0.0,
                "final_selection": "initial_self",
                "root_action_stats": {},
                "llm_calls": self.llm.call_count,
            }

        correct_count = 0

        for rollout_idx in range(self.cfg.num_rollouts):
            logger.debug("Rollout %d/%d", rollout_idx + 1, self.cfg.num_rollouts)

            node = self._select(root)
            if not node.is_terminal and node.step_count < self.cfg.max_depth:
                node = self._expand(node, problem, reference_answer)

            terminal_reward, trajectory, verifier = self._simulate(
                node, problem, reference_answer
            )
            mean_progress = self._mean_path_progress(node)
            search_return = (
                terminal_reward
                + self.cfg.progress_weight * mean_progress
            )
            self._backpropagate(node, search_return)

            if terminal_reward > 0:
                correct_count += 1
                if node.successful_trajectory is None:
                    node.successful_trajectory = copy.deepcopy(trajectory)
                    node.successful_verifier = copy.deepcopy(verifier)

            if rollout_callback is not None:
                rollout_callback(
                    rollout_idx,
                    terminal_reward,
                    node,
                    correct_count,
                    mean_progress=mean_progress,
                    search_return=search_return,
                )
        principal_leaf = self._principal_leaf(root)
        best_trajectory = None
        best_reward = 0.0
        best_verifier = None
        final_selection = "principal_visits"

        # Reuse a correct continuation from this exact principal leaf when possible.
        if principal_leaf.successful_trajectory is not None:
            best_trajectory = copy.deepcopy(principal_leaf.successful_trajectory)
            best_reward = 1.0
            best_verifier = copy.deepcopy(principal_leaf.successful_verifier)
        else:
            # The pedagogical path is already fixed by visits. This one autonomous
            # completion does not choose among teaching actions; it only materializes
            # a complete SFT trajectory from the selected leaf.
            final_reward, final_traj, final_verifier = self._simulate(
                principal_leaf, problem, reference_answer
            )
            if final_reward > 0:
                best_trajectory = final_traj
                best_reward = 1.0
                best_verifier = final_verifier
                final_selection = "principal_visits_fresh_rollout"
            else:
                # Robust fallback: use the successful searched node with the largest
                # visit count. This is still purely tree-statistical and never uses
                # teacher count, evaluator quality, or a teacher reranker.
                fallback = self._most_visited_successful_node(root)
                if fallback is not None:
                    best_trajectory = copy.deepcopy(fallback.successful_trajectory)
                    best_reward = 1.0
                    best_verifier = copy.deepcopy(fallback.successful_verifier)
                    principal_leaf = fallback
                    final_selection = "successful_visit_fallback"

        best_action_sequence = ["initial_self"] + self._tree_action_sequence(principal_leaf)
        if best_trajectory and best_trajectory[-1].get("source") == "rollout":
            best_action_sequence.append("rollout_self")

        return {
            "problem": problem,
            "reference_answer": reference_answer,
            "best_trajectory": best_trajectory,
            "best_reward": best_reward,
            "best_verifier": best_verifier,
            "correct_rate": correct_count / self.cfg.num_rollouts,
            "total_rollouts": self.cfg.num_rollouts,
            "tree_depth": self._max_depth(root),
            "initial_self_step": root_info["initial_self_step"],
            "initial_self_correct": False,
            "initial_self_wrong_final": root_info["initial_self_wrong_final"],
            "best_action_sequence": best_action_sequence,
            "teacher_turns": self._count_teacher_turns(best_trajectory),
            "mean_progress": self._mean_path_progress(principal_leaf),
            "final_selection": final_selection,
            "root_action_stats": self._node_action_stats(root),
            "llm_calls": self.llm.call_count,
        }

    # ── Root ────────────────────────────────────────────────────────────────

    def _initialize_root(
        self,
        problem: str,
        reference_answer: str,
    ) -> tuple[MCTSNode, dict]:
        first = self.llm.student_step(problem, [])
        history = [{
            "role": "student",
            "content": first.text,
            "source": "initial_self",
            "truncated": first.truncated,
        }]

        initial_self_correct = False
        initial_self_wrong_final = False
        verifier = None

        if _has_final_answer(first.text) and not first.truncated:
            verifier = self.llm.check_solution(problem, history, reference_answer)
            initial_self_correct = bool(verifier["correct"])
            initial_self_wrong_final = not initial_self_correct

        # ✅ 修复：截断不应导致立即终止，应允许后续扩展尝试
        # 只有正确答案才应终止（无需继续搜索）
        root = MCTSNode(
            conversation_history=history,
            step_count=1,
            is_terminal=initial_self_correct,  # 仅当初始答案正确时终止
            is_correct=True if initial_self_correct else None,
            terminal_verifier=verifier if initial_self_correct else None,
        )

        # Wrong FINAL ANSWER remains repairable by later teaching actions.
        if initial_self_wrong_final and not first.truncated:
            root.is_terminal = False

        # 截断的初始步骤仍然可以扩展（通过重采样可能得到非截断后继）
        # 但在返回信息中标记，以便调试和分析
        if first.truncated:
            logger.warning("Initial student step was truncated, allowing expansion anyway")

        return root, {
            "initial_self_step": first.text,
            "initial_self_correct": initial_self_correct,
            "initial_self_wrong_final": initial_self_wrong_final,
        }

    # ── Select / Expand ────────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while (
            not node.is_terminal
            and node.step_count < self.cfg.max_depth
            and node.is_fully_expanded(self.cfg.actions)
            and node.children
        ):
            node = node.best_child(self.cfg.exploration_c)
        return node

    def _expand(
        self,
        node: MCTSNode,
        problem: str,
        reference_answer: str,
    ) -> MCTSNode:
        untried = [a for a in self.cfg.actions if a not in node.children]
        if not untried:
            return node

        # Equal treatment: no forced SELF-first expansion.
        action = random.choice(untried)

        before_history = copy.deepcopy(node.conversation_history)
        new_history = copy.deepcopy(before_history)
        teacher_feedback = None
        teacher_truncated = False

        if action != "self":
            teacher = self.llm.teacher_act(problem, before_history, action)
            teacher_feedback = teacher.text
            teacher_truncated = teacher.truncated
            new_history.append({
                "role": f"teacher_{action}",
                "content": teacher.text,
                "source": "tree_teacher",
                "truncated": teacher.truncated,
            })

        student = self.llm.student_step(problem, new_history)
        new_history.append({
            "role": "student",
            "content": student.text,
            "source": "tree_step",
            "action": action,
            "truncated": student.truncated,
        })

        transition_eval = self.llm.evaluate_progress(
            problem,
            reference_answer,
            before_history=before_history,
            action_type=action,
            teacher_feedback=teacher_feedback,
            student_response=student.text,
            truncated=teacher_truncated or student.truncated,
        )

        # Partial steps are repairable by later reasoning, not terminal failures.
        is_terminal = False
        is_correct = None
        terminal_verifier = None

        # If the student emits FINAL ANSWER, verify immediately. A correct answer
        # closes the branch; a wrong answer stays open so later critique/hint can repair it.
        if _has_final_answer(student.text) and not student.truncated:
            terminal_verifier = self.llm.check_solution(
                problem, new_history, reference_answer
            )
            is_correct = bool(terminal_verifier["correct"])
            if is_correct:
                is_terminal = True

        child = MCTSNode(
            conversation_history=new_history,
            step_count=node.step_count + 1,
            transition_eval=transition_eval,
            transition_progress=float(transition_eval["normalized_progress"]),
            is_terminal=is_terminal,
            is_correct=is_correct,
            terminal_verifier=terminal_verifier if is_correct else None,
        )
        node.add_child(action, child)
        return child

    # ── Simulation ─────────────────────────────────────────────────────────

    def _simulate(
        self,
        node: MCTSNode,
        problem: str,
        reference_answer: str,
    ) -> tuple[float, list[dict], dict]:
        if node.is_terminal:
            if node.conversation_history[-1].get("truncated"):
                verifier = {"correct": False, "reason": "unfinished final transition"}
                return self.cfg.wrong_reward, copy.deepcopy(node.conversation_history), verifier

            if node.is_correct is True and node.terminal_verifier is not None:
                return (
                    self.cfg.correct_reward,
                    copy.deepcopy(node.conversation_history),
                    copy.deepcopy(node.terminal_verifier),
                )

            verifier = self.llm.check_solution(
                problem, node.conversation_history, reference_answer
            )
            node.is_correct = bool(verifier["correct"])
            node.terminal_verifier = verifier
            reward = self.cfg.correct_reward if node.is_correct else self.cfg.wrong_reward
            return reward, copy.deepcopy(node.conversation_history), verifier

        rollout = self.llm.student_rollout(problem, node.conversation_history)
        full_trajectory = copy.deepcopy(node.conversation_history)
        full_trajectory.append({
            "role": "student",
            "content": rollout.text,
            "source": "rollout",
            "action": "self",
            "truncated": rollout.truncated,
        })

        if rollout.truncated or not _has_final_answer(rollout.text):
            verifier = {
                "correct": False,
                "reason": "rollout truncated" if rollout.truncated else "rollout missing FINAL ANSWER",
            }
            return self.cfg.wrong_reward, full_trajectory, verifier

        verifier = self.llm.check_solution(problem, full_trajectory, reference_answer)
        reward = self.cfg.correct_reward if verifier["correct"] else self.cfg.wrong_reward
        return reward, full_trajectory, verifier

    # ── Return / Backup ────────────────────────────────────────────────────

    def _mean_path_progress(self, node: MCTSNode) -> float:
        values = []
        current = node
        while current.parent is not None:
            values.append(current.transition_progress)
            current = current.parent
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _backpropagate(self, node: MCTSNode, search_return: float) -> None:
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_return += search_return
            current = current.parent

    # ── Final tree-statistical policy ──────────────────────────────────────

    def _principal_leaf(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children:
            visited = [ch for ch in node.children.values() if ch.visit_count > 0]
            if not visited:
                break
            # Primary: visit count. Secondary: learned Q. Any exact remaining tie
            # follows random expansion insertion order, so no action gets a fixed bias.
            node = max(visited, key=lambda ch: (ch.visit_count, ch.q_value))
            if node.is_terminal or node.step_count >= self.cfg.max_depth:
                break
        return node

    def _most_visited_successful_node(self, root: MCTSNode) -> Optional[MCTSNode]:
        successes = []

        def walk(n: MCTSNode):
            if n.successful_trajectory is not None:
                successes.append(n)
            for child in n.children.values():
                walk(child)

        walk(root)
        if not successes:
            return None
        return max(successes, key=lambda n: (n.visit_count, n.q_value, n.step_count))

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _count_teacher_turns(trajectory: Optional[list[dict]]) -> int:
        if not trajectory:
            return 0
        return sum(
            1 for t in trajectory
            if t.get("role", "").startswith("teacher_")
        )

    def _tree_action_sequence(self, leaf: MCTSNode) -> list[str]:
        actions = []
        node = leaf
        while node.parent is not None:
            actions.append(node.action_from_parent)
            node = node.parent
        actions.reverse()
        return actions

    def _node_action_stats(self, node: MCTSNode) -> dict:
        return {
            action: {
                "visits": child.visit_count,
                "q_value": child.q_value,
                "transition_progress": child.transition_progress,
                "transition_eval": child.transition_eval,
            }
            for action, child in node.children.items()
        }

    def _max_depth(self, node: MCTSNode, depth: int = 0) -> int:
        if not node.children:
            return depth
        return max(self._max_depth(child, depth + 1) for child in node.children.values())
