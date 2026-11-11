"""Offline reliability tests. These do not measure DeepSeek/public-test accuracy."""
import asyncio
import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import tool

import hw3


ANSWERS = {
    "sales_team": "Our account manager provides expert technical advice and reliable support.",
    "working_conditions": "Staff receive fair pay and managers resolve safety concerns promptly.",
    "competitive_edge": "Their specialized process is difficult for competitors to replicate.",
}


def fixture_tools(*, unavailable_first=False, broken_topic=None):
    calls = []

    @tool
    def interview(role: str, topic: str) -> str:
        """Return synthetic interview evidence for a reliability test."""
        calls.append((role, topic))
        if topic == broken_topic:
            raise RuntimeError("Simulated tool failure")
        if unavailable_first and role == next(
            plan[1] for plan in hw3.INTERVIEW_PLAN.values() if plan[0] == topic
        ):
            return "I cannot comment on that."
        return ANSWERS[topic]

    return [interview], calls


def response_for(messages):
    records = json.loads(messages[1].content)["evidence"]
    return SimpleNamespace(content=json.dumps({"decisions": [
        {"point": item["point"], "verdict": "PASS", "evidence_quote": item["answer"],
         "reason": "The interview provides direct support for the specified criterion."}
        for item in records
    ]}))


class ControlledModel:
    def __init__(self, invalid_first=False, fail=False):
        self.calls = 0
        self.invalid_first, self.fail = invalid_first, fail

    async def ainvoke(self, messages):
        self.calls += 1
        if self.fail:
            raise ValueError("Simulated model failure")
        if self.invalid_first and self.calls == 1:
            return SimpleNamespace(content='{"decisions": []}')
        return response_for(messages)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {
            p: {"role": role, "topic": topic, "answer": ANSWERS[topic]}
            for p, (topic, role, _) in hw3.INTERVIEW_PLAN.items()
        }
        self.payload = {"decisions": [
            {"point": p, "verdict": "PASS", "evidence_quote": e["answer"],
             "reason": "The specific interview supports this rule."}
            for p, e in self.evidence.items()
        ]}

    def test_valid_decisions(self):
        actual = hw3._validate_decisions(json.dumps(self.payload), self.evidence)
        self.assertEqual(set(actual), {4, 7, 11})

    def test_fabricated_quote_rejected(self):
        self.payload["decisions"][0]["evidence_quote"] = "Invented support that the witness never supplied."
        with self.assertRaises(ValueError):
            hw3._validate_decisions(json.dumps(self.payload), self.evidence)

    def test_quote_from_other_point_rejected(self):
        self.payload["decisions"][0]["evidence_quote"] = ANSWERS["working_conditions"]
        with self.assertRaises(ValueError):
            hw3._validate_decisions(json.dumps(self.payload), self.evidence)

    def test_duplicate_point_rejected(self):
        self.payload["decisions"][1] = copy.deepcopy(self.payload["decisions"][0])
        with self.assertRaises(ValueError):
            hw3._validate_decisions(json.dumps(self.payload), self.evidence)

    def test_invalid_verdict_rejected(self):
        self.payload["decisions"][0]["verdict"] = "UNCERTAIN"
        with self.assertRaises(ValueError):
            hw3._validate_decisions(json.dumps(self.payload), self.evidence)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_informed_interviews(self):
        tools, calls = fixture_tools()
        evidence, errors = await hw3._gather_evidence(tools)
        self.assertEqual(set(evidence), {4, 7, 11})
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 3)

    async def test_backups_stay_within_budget(self):
        tools, calls = fixture_tools(unavailable_first=True)
        evidence, errors = await hw3._gather_evidence(tools)
        self.assertEqual(len(evidence), 3)
        self.assertFalse(errors)
        self.assertEqual(len(calls), 6)

    async def test_partial_tool_failure_preserves_other_points(self):
        tools, _ = fixture_tools(broken_topic="working_conditions")
        result = await hw3.analyze_companies({"model": ControlledModel()}, {"renamed": tools})
        self.assertEqual(result["renamed"]["points"], {4: "PASS", 11: "PASS"})
        self.assertIn("Point 7: no usable interview evidence", result["renamed"]["summary"])

    async def test_repair_reuses_original_interviews(self):
        tools, calls = fixture_tools()
        model = ControlledModel(invalid_first=True)
        result = await hw3.analyze_companies({"model": model}, {"unseen_name": tools})
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result["unseen_name"]["points"]), 3)

    async def test_model_failure_does_not_crash_run(self):
        tools, _ = fixture_tools()
        result = await hw3.analyze_companies({"model": ControlledModel(fail=True)}, {"x": tools})
        self.assertEqual(result["x"]["points"], {})
        self.assertIn("Analysis unavailable", result["x"]["summary"])

    async def test_company_failure_is_isolated(self):
        tools, _ = fixture_tools()
        result = await hw3.analyze_companies({"model": ControlledModel()}, {"bad": [], "good": tools})
        self.assertEqual(result["bad"]["points"], {})
        self.assertEqual(len(result["good"]["points"]), 3)

    async def test_missing_key_returns_entries(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            agent = hw3.build_agent()
        result = await hw3.analyze_companies(agent, {"one": [], "two": []})
        self.assertEqual(set(result), {"one", "two"})
        self.assertTrue(all(x["points"] == {} for x in result.values()))


if __name__ == "__main__":
    unittest.main()
