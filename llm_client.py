"""
LLM Client
==========
Single-model interface for clean EduMCTS.

The evaluator is deliberately narrow: it does not estimate a separate value
function and does not score multiple DCS dimensions. It only judges the
realized change in the student's reasoning after one control action.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import logging
import re
import time
from typing import Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

from config import ModelConfig, MODEL_CFG

logger = logging.getLogger(__name__)


TEACHER_SYSTEM_PROMPT = """You are an expert mathematics/reasoning tutor.
You supervise an already-started student solution.

Your intervention must be LOCAL, concise, and useful for the CURRENT reasoning state.
Never provide the final answer or a full replacement solution.
Do not praise the student. Do not add motivational filler.
Do not invent an error that is not actually present.
Return only the intervention text."""


STUDENT_STEP_SYSTEM_PROMPT = """You are the student solving the problem.
Continue from the existing reasoning history with exactly ONE atomic reasoning step.

Rules:
- Advance only one local subgoal, inference, check, or computation.
- Do not restart or summarize the whole solution.
- Prefer <= 120 words unless equations require slightly more space.
- If a tutor intervention is present, use it as guidance but do the reasoning yourself.
- If this single step genuinely completes the whole problem, end with:
  FINAL ANSWER: <answer or proved statement>
- Otherwise do NOT output FINAL ANSWER.
Return only the next reasoning step."""


ROLLOUT_SYSTEM_PROMPT = """You are the student completing a reasoning problem autonomously.
Continue from the supplied reasoning history and finish the remaining solution in ONE response.

Rules:
- Do not restart from the beginning or repeat already-established work.
- Resolve any visible unresolved issue before relying on it.
- Give enough reasoning for the final conclusion to be justified.
- Solve every requested subpart.
- End with exactly:
  FINAL ANSWER: <complete answer or proved statement>
Return only the continuation and final answer."""


PROGRESS_EVALUATOR_SYSTEM_PROMPT = r"""You are a STRICT evaluator of one realized student-reasoning transition.
You are NOT evaluating how nice the teacher message sounds. You judge whether the student's NEW reasoning is better than the BEFORE state after the chosen control action.

You will receive:
1. the original problem;
2. a reference answer/conclusion for checking correctness (never shown to the student);
3. the reasoning BEFORE the action;
4. the chosen action: self / hint / critique / socratic;
5. the teacher intervention, if any;
6. the student's NEW one-step response;
7. whether generation was truncated.

Return ONE integer progress score in {-2,-1,0,1,2}.

+2 = major positive change: fixes a key error, resolves a real bottleneck, or adds a decisive correct insight that materially improves the solution state.
+1 = modest but real positive change: a correct, useful next inference/check/computation that advances the solution.
 0 = essentially neutral: harmless bookkeeping, restatement, very weak progress, or an intervention that was unnecessary and produced no meaningful improvement.
-1 = meaningful deterioration: introduces a nontrivial gap, confusion, unsupported move, or follows a poorly targeted intervention in a way that weakens the reasoning state.
-2 = major deterioration: concrete mathematical/logical error, contradiction, compounding an existing key error, or a teacher intervention that effectively gives away the solution/final answer so the student merely copies rather than reasons.

Critical rules:
- Judge BEFORE -> NEW STUDENT RESPONSE, not the teacher text in isolation.
- A good-looking hint that leads to wrong reasoning must not receive a positive score.
- A critique that invents a nonexistent flaw should be non-positive.
- SELF can receive +1/+2 when autonomous reasoning genuinely progresses.
- Do not reward verbosity, confidence, or surface similarity to the reference answer.
- If truncated=true, score -2.
- Be conservative: +2 should be rare.

Return ONLY valid JSON:
{"progress": 0, "reason": "one short diagnostic sentence"}"""


SOLUTION_VERIFIER_SYSTEM_PROMPT = """You are a strict verifier of a complete mathematical/logical solution.
Decide whether the supplied trajectory fully and correctly answers the ORIGINAL problem.

Rules:
- Check the reasoning, not only the text after FINAL ANSWER.
- For proof problems, a bare conclusion is not sufficient: the proof must be logically valid.
- For multi-part problems, every requested part must be correctly solved.
- Reject a trajectory with a material unresolved contradiction or invalid step.
- Equivalent mathematical forms are acceptable.
- Teacher/reflection text may guide reasoning but does not excuse a missing or invalid solution.
- Marked partial steps must be completed or repaired by later reasoning.
Keep reason to one plain-text sentence of at most 30 words, without LaTeX.
Return ONLY valid JSON:
{"correct": true, "reason": "brief reason"}"""


@dataclass
class Generation:
    text: str
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _strip_markdown_fence(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else raw.strip()


def _sanitize_latex_escapes(raw: str) -> str:
    return re.sub(r'\\(?!["\\\/bfnrtu])', r'\\\\', raw)


def _parse_json(raw: str) -> dict:
    cleaned = _strip_markdown_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_sanitize_latex_escapes(cleaned))


def _clamp_progress(x) -> int:
    try:
        return max(-2, min(2, int(x)))
    except (TypeError, ValueError):
        return 0


class VerifierResponseError(RuntimeError):
    """Invalid verifier output is an execution failure, not a wrong solution."""
    def __init__(self, diagnostics):
        super().__init__("Verifier JSON invalid/truncated after format retries")
        self.diagnostics = diagnostics


class LLMClient:
    def __init__(self, cfg: ModelConfig = MODEL_CFG):
        cfg.validate()
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package is required: pip install openai")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self._call_count = 0
        self.verifier_diagnostics = []

    @property
    def call_count(self) -> int:
        return self._call_count

    def _chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Generation:
        last_error = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.cfg.timeout,
                )
                self._call_count += 1

                # ✅ 修复：检查 API 返回类型，避免 'str' object has no attribute 'choices'
                if isinstance(resp, str):
                    raise ValueError(f"API returned string instead of ChatCompletion: {resp[:200]}")
                if not hasattr(resp, 'choices') or not resp.choices:
                    raise ValueError(f"API response missing choices: {type(resp).__name__}")

                choice = resp.choices[0]
                return Generation(
                    text=(choice.message.content or "").strip(),
                    finish_reason=getattr(choice, "finish_reason", None),
                )
            except Exception as e:
                last_error = e
                logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "(No previous reasoning. This is the first student step.)"
        chunks = []
        for turn in history:
            role = turn.get("role", "unknown").upper()
            partial = " [PARTIAL: complete or repair before proceeding]" if turn.get("truncated") else ""
            chunks.append(f"[{role}]{partial}\n{turn.get('content', '')}")
        return "\n\n".join(chunks)

    def teacher_act(
        self,
        problem: str,
        conversation_history: list[dict],
        action_type: str,
    ) -> Generation:
        instructions = {
            "hint": """HINT:
Identify the student's current bottleneck and give ONE directional clue.
Do not state the next full derivation, final formula, or final answer.
If the student is already progressing well, point to the next useful consideration rather than restating prior work.""",
            "critique": """CRITIQUE:
Inspect the existing reasoning for a CONCRETE mathematical/logical error, unsupported assumption, contradiction, or invalid inference.
- If a concrete error exists: identify exactly ONE most important error and explain what must be reconsidered, without giving the corrected full solution.
- If NO definite error exists: do NOT invent one and do NOT say the reasoning is wrong. Instead identify ONE genuinely error-prone point, hidden assumption, boundary case, or calculation that should be checked.""",
            "socratic": """SOCRATIC:
Ask exactly ONE targeted question that makes the student inspect the current bottleneck, derive a missing relation, or verify a risky assumption.
The question must be answerable from the problem/current reasoning.
Do not answer the question yourself and do not reveal the final answer.""",
        }
        if action_type not in instructions:
            raise ValueError(f"Unknown teacher action: {action_type}")

        user = (
            f"PROBLEM:\n{problem}\n\n"
            f"CURRENT REASONING:\n{self._format_history(conversation_history)}\n\n"
            f"REQUIRED ACTION:\n{instructions[action_type]}"
        )
        result = self._chat(
            TEACHER_SYSTEM_PROMPT,
            user,
            temperature=self.cfg.teacher_temperature,
            max_tokens=self.cfg.teacher_max_tokens,
        )
        for attempt in range(self.cfg.generation_repair_attempts):
            if not result.truncated:
                break
            result = self._chat(
                TEACHER_SYSTEM_PROMPT,
                user + "\n\nRetry more concisely in at most 80 words.",
                temperature=self.cfg.teacher_temperature,
                max_tokens=self.cfg.teacher_max_tokens * 2 ** (attempt + 1),
            )
        return result

    def student_step(
        self,
        problem: str,
        conversation_history: list[dict],
    ) -> Generation:
        user = (
            f"PROBLEM:\n{problem}\n\n"
            f"REASONING / TUTOR HISTORY:\n{self._format_history(conversation_history)}\n\n"
            "Produce the next atomic reasoning step."
        )
        result = self._chat(
            STUDENT_STEP_SYSTEM_PROMPT,
            user,
            temperature=self.cfg.student_step_temperature,
            max_tokens=self.cfg.student_step_max_tokens,
        )
        for attempt in range(self.cfg.generation_repair_attempts):
            if not result.truncated:
                break
            result = self._chat(
                STUDENT_STEP_SYSTEM_PROMPT,
                user + "\n\nRetry with ONE complete step in <= 80 words.",
                temperature=self.cfg.student_step_temperature,
                max_tokens=self.cfg.student_step_max_tokens * 2 ** (attempt + 1),
            )
        return result

    # Backward-compatible helper.
    def student_respond(self, problem: str, conversation_history: list[dict]) -> str:
        return self.student_step(problem, conversation_history).text

    def student_rollout(
        self,
        problem: str,
        conversation_history: list[dict],
    ) -> Generation:
        user = (
            f"PROBLEM:\n{problem}\n\n"
            f"REASONING / TUTOR HISTORY SO FAR:\n{self._format_history(conversation_history)}\n\n"
            "Continue autonomously from exactly this point and finish the entire remaining solution."
        )
        return self._chat(
            ROLLOUT_SYSTEM_PROMPT,
            user,
            temperature=self.cfg.rollout_temperature,
            max_tokens=self.cfg.rollout_max_tokens,
        )

    def evaluate_progress(
        self,
        problem: str,
        reference_answer: str,
        before_history: list[dict],
        action_type: str,
        student_response: str,
        *,
        teacher_feedback: Optional[str] = None,
        truncated: bool = False,
    ) -> dict:
        """Score one realized tree transition on {-2,-1,0,+1,+2}."""
        if truncated:
            return {
                "progress": -2,
                "normalized_progress": -1.0,
                "reason": "generation was truncated",
                "parse_error": False,
            }

        user = (
            f"PROBLEM:\n{problem}\n\n"
            f"REFERENCE ANSWER / CONCLUSION (for evaluator only):\n{reference_answer}\n\n"
            f"BEFORE REASONING:\n{self._format_history(before_history)}\n\n"
            f"CONTROL ACTION: {action_type}\n\n"
            f"TEACHER INTERVENTION:\n{teacher_feedback if teacher_feedback else '(none; autonomous Self)'}\n\n"
            f"NEW STUDENT RESPONSE:\n{student_response}\n\n"
            f"TRUNCATED: {str(bool(truncated)).lower()}"
        )
        raw = self._chat(
            PROGRESS_EVALUATOR_SYSTEM_PROMPT,
            user,
            temperature=self.cfg.evaluator_temperature,
            max_tokens=self.cfg.evaluator_max_tokens,
        ).text

        parse_error = False
        try:
            data = _parse_json(raw)
        except Exception:
            parse_error = True
            data = {}
            m = re.search(r'"progress"\s*:\s*(-?\d+)', raw)
            if m:
                data["progress"] = int(m.group(1))
            rm = re.search(r'"reason"\s*:\s*"(.*?)"\s*[},]', raw, re.DOTALL)
            if rm:
                data["reason"] = rm.group(1)[:500]

        progress = _clamp_progress(data.get("progress", 0))
        # Parse failures are deliberately neutral rather than silently positive.
        if parse_error and "progress" not in data:
            progress = 0

        return {
            "progress": progress,
            "normalized_progress": progress / 2.0,
            "reason": str(data.get("reason", "parse fallback" if parse_error else ""))[:500],
            "parse_error": parse_error,
        }

    def check_solution(
        self,
        problem: str,
        conversation_history: list[dict],
        reference_answer: str,
    ) -> dict:
        user = (
            f"ORIGINAL PROBLEM:\n{problem}\n\n"
            f"REFERENCE ANSWER / REFERENCE CONCLUSION:\n{reference_answer}\n\n"
            f"CANDIDATE TRAJECTORY:\n{self._format_history(conversation_history)}"
        )
        diagnostics = []
        for attempt in range(self.cfg.verifier_format_retries + 1):
            retry_prompt = "" if attempt == 0 else (
                '\nRe-evaluate and return ONLY one JSON object with boolean correct '
                'and a short plain-text reason. No markdown or LaTeX.'
            )
            response = self._chat(
                SOLUTION_VERIFIER_SYSTEM_PROMPT, user + retry_prompt,
                temperature=self.cfg.verifier_temperature,
                max_tokens=self.cfg.verifier_max_tokens * 2 ** attempt,
            )
            try:
                if response.truncated:
                    raise ValueError("Verifier output reached token limit")
                data = _parse_json(response.text)
                if (not isinstance(data, dict) or type(data.get("correct")) is not bool
                        or not isinstance(data.get("reason"), str)):
                    raise ValueError("Expected boolean correct and string reason")
                return {"correct": data["correct"], "reason": data["reason"][:500],
                        "format_retries": attempt, "parse_error": False}
            except (ValueError, TypeError) as exc:
                diagnostic = {
                    "problem_hash": hashlib.sha256(problem.encode()).hexdigest()[:16],
                    "attempt": attempt + 1, "finish_reason": response.finish_reason,
                    "error": str(exc), "raw_response": response.text,
                }
                diagnostics.append(diagnostic)
                self.verifier_diagnostics.append(diagnostic)
                logger.warning("Verifier format failure: problem=%s attempt=%d finish_reason=%s error=%s raw=%r",
                               diagnostic["problem_hash"], attempt + 1, response.finish_reason,
                               str(exc), response.text[:500])
        raise VerifierResponseError(diagnostics)

    def check_answer(self, problem: str, student_answer: str, reference_answer: str) -> bool:
        history = [{"role": "student", "content": f"FINAL ANSWER: {student_answer}"}]
        return self.check_solution(problem, history, reference_answer)["correct"]
