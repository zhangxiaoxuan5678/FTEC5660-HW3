"""Run the actual DeepSeek agent and update the README with measured results.

The public answer key is read only AFTER hw3.py finishes. It is never supplied
to the agent. A failed/incomplete run cannot be reported as a successful test.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import hw3


def main() -> int:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    hw3.load_env_file()
    if hw3.build_agent()["model"] is None:
        print("Configure a valid DEEPSEEK_API_KEY in .env before evaluation.", file=sys.stderr)
        return 2
    # Remove a previous output so a failed subprocess cannot reuse stale results.
    output = root / "results.csv"
    output.unlink(missing_ok=True)
    start = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "hw3.py", "--data-folder", "public_test"],
            timeout=1740, check=False,
        )
    except subprocess.TimeoutExpired:
        print("Evaluation exceeded its time limit; README was not updated.", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - start
    if result.returncode != 0 or not output.is_file():
        print("Evaluation failed; README was not updated.", file=sys.stderr)
        return 2
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [p.name for p in hw3.company_folders(root / "public_test")]
    if len(rows) != len(names) or {r["company"] for r in rows} != set(names):
        raise SystemExit("Incomplete company coverage; README was not updated.")
    analyses = {}
    for row in rows:
        points = dict(item.split(":", 1) for item in row["points"].split())
        normalized = hw3.normalize_points(points, hw3.CHOSEN_POINTS)
        if any(v is None for v in normalized.values()):
            raise SystemExit("Missing model verdicts; fix API/tool errors before reporting accuracy.")
        analyses[row["company"]] = {"points": normalized}
    truth = hw3.read_ground_truth(root / "public_test")
    score = hw3.balanced_point_accuracy(analyses, truth, names, hw3.CHOSEN_POINTS)
    correct = sum(hw3.point_errors(analyses[n]["points"], truth[n])[0] for n in names)
    lines = [
        f"Actual model: `{hw3.MODEL_NAME}`. Companies: {len(names)}. "
        f"Selected-point accuracy: **{correct}/{len(names) * 3}**. "
        f"Balanced point score: **{score:.2%}**. Runtime: **{elapsed:.1f} seconds**.",
        "", "These are measured public-test results; private-test accuracy may differ.", "",
        f"Interview calls per company: {min(int(r['interviews_used']) for r in rows)}"
        f" to {max(int(r['interviews_used']) for r in rows)}.", "",
    ]
    wrong = [r for r in rows if r["point_errors"]]
    if wrong:
        lines += ["| Company | Disagreement | Agent explanation |", "| --- | --- | --- |"]
        for row in wrong:
            cells = [row["company"], row["point_errors"], row["summary"]]
            lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
        lines += ["", "The table records the exact disagreements and the agent's evidence. "
                  "Their causes require manual review; the model's explanation alone does not establish the cause."]
    else:
        lines += ["No selected-point verdict disagreed with the public answers in this run."]
    versions = ", ".join(f"{p} {importlib.metadata.version(p)}" for p in ("langchain", "langchain-core", "langchain-deepseek"))
    lines += ["", f"Environment: Python {sys.version.split()[0]}; {versions}."]
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    begin, end = "<!-- PUBLIC_RESULTS_START -->", "<!-- PUBLIC_RESULTS_END -->"
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    readme.write_text(before + begin + "\n" + "\n".join(lines) + "\n" + end + after, encoding="utf-8")
    (root / "public_results.csv").write_bytes(output.read_bytes())
    print("Updated README.md and public_results.csv with actual DeepSeek results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
