from unittest.mock import patch

from config import MCTSConfig
from harvester import trajectory_to_cot
from mcts import EduMCTS, MCTSNode


class Gen:
    def __init__(self, text, truncated=False):
        self.text = text
        self._truncated = truncated

    @property
    def truncated(self):
        return self._truncated


class MockLLM:
    def __init__(self):
        self.call_count = 0
        self.step_calls = 0
        self.rollout_calls = 0
        self.teacher_calls = 0
        self.eval_calls = 0

    def student_step(self, problem, history):
        self.step_calls += 1
        if self.step_calls == 1:
            return Gen("Let me derive the key relation first.")
        return Gen("This gives the next useful relation.")

    def teacher_act(self, problem, history, action):
        self.teacher_calls += 1
        return Gen(f"{action}: check the local bottleneck.")

    def student_rollout(self, problem, history):
        self.rollout_calls += 1
        return Gen("Continuing from here, the remaining derivation gives 42.\nFINAL ANSWER: 42")

    def evaluate_progress(
        self, problem, reference_answer, before_history, action_type,
        student_response, teacher_feedback=None, truncated=False
    ):
        self.eval_calls += 1
        score = 2 if action_type == "self" else 1
        return {
            "progress": score,
            "normalized_progress": score / 2.0,
            "reason": "mock",
            "parse_error": False,
        }

    def check_solution(self, problem, history, reference_answer):
        text = "\n".join(t["content"] for t in history)
        return {"correct": "FINAL ANSWER: 42" in text, "reason": "mock"}


def test_root_is_initial_self_and_needs_no_evaluator():
    llm = MockLLM()
    mcts = EduMCTS(llm, MCTSConfig(num_rollouts=1, max_depth=3))
    root, _ = mcts._initialize_root("p", "42")

    assert root.step_count == 1
    assert root.conversation_history[0]["role"] == "student"
    assert root.conversation_history[0]["source"] == "initial_self"
    assert llm.teacher_calls == 0
    assert llm.eval_calls == 0
    assert root.transition_eval is None


def test_self_and_teacher_are_equal_untried_actions():
    llm = MockLLM()
    mcts = EduMCTS(llm, MCTSConfig(num_rollouts=1, max_depth=3))
    root, _ = mcts._initialize_root("p", "42")

    with patch("mcts.random.choice", return_value="self"):
        child_self = mcts._expand(root, "p", "42")
    assert child_self.action_from_parent == "self"
    assert llm.teacher_calls == 0
    assert child_self.transition_progress == 1.0

    root2, _ = mcts._initialize_root("p", "42")
    with patch("mcts.random.choice", return_value="critique"):
        child_teacher = mcts._expand(root2, "p", "42")
    assert child_teacher.action_from_parent == "critique"
    assert llm.teacher_calls == 1
    assert child_teacher.transition_progress == 0.5


def test_rollout_is_one_call_appended_and_not_progress_evaluated():
    llm = MockLLM()
    mcts = EduMCTS(llm, MCTSConfig(num_rollouts=1, max_depth=3))
    root, _ = mcts._initialize_root("p", "42")

    reward, trajectory, verifier = mcts._simulate(root, "p", "42")

    assert reward == 1.0
    assert verifier["correct"] is True
    assert llm.rollout_calls == 1
    assert llm.eval_calls == 0
    assert trajectory[-1]["source"] == "rollout"
    assert "FINAL ANSWER: 42" in trajectory[-1]["content"]


def test_standard_uct_uses_single_q_value():
    parent = MCTSNode(visit_count=10)
    a = MCTSNode(visit_count=5, total_return=4.0)
    b = MCTSNode(visit_count=5, total_return=2.0)
    parent.add_child("a", a)
    parent.add_child("b", b)
    assert parent.best_child(c=0.0) is a
    assert a.q_value == 0.8


def test_backpropagates_one_unified_return():
    llm = MockLLM()
    mcts = EduMCTS(llm, MCTSConfig())
    root = MCTSNode()
    child = MCTSNode(parent=root)
    mcts._backpropagate(child, 1.125)
    assert child.visit_count == 1 and root.visit_count == 1
    assert child.total_return == 1.125 and root.total_return == 1.125


def test_principal_path_uses_visits_not_teacher_count():
    llm = MockLLM()
    mcts = EduMCTS(llm, MCTSConfig())
    root = MCTSNode()
    self_child = MCTSNode(visit_count=8, total_return=5.0)
    teacher_child = MCTSNode(visit_count=3, total_return=3.0)
    root.add_child("self", self_child)
    root.add_child("critique", teacher_child)
    assert mcts._principal_leaf(root) is self_child


def test_harvester_has_no_step_labels_and_never_fabricates_answer():
    trajectory = [
        {"role": "student", "content": "Start reasoning."},
        {"role": "teacher_critique", "content": "Verify the assumption."},
        {"role": "student", "content": "I verify it and finish.\nFINAL ANSWER: 42"},
    ]
    text = trajectory_to_cot("p", trajectory, "42")
    assert "[Step" not in text
    assert '<reflection type="critique">' in text
    assert "FINAL ANSWER: 42" in text

    unfinished = trajectory_to_cot(
        "p", [{"role": "student", "content": "Still working."}], "42"
    )
    assert "FINAL ANSWER" not in unfinished
