"""Evaluation script for the travel planning agent.

Runs each test case through the agent, collects metrics, and outputs a summary.
"""
import json
import os
import sys
import re
import time
from pathlib import Path

# Allow running as `python evaluation/react_eval.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from agents.react_agent import run_agent
from prompts import SYSTEM_PROMPT
from config import OPENROUTER_API_KEY, MODEL_NAME, JUDGE_MODEL_NAME

EVAL_DIR = Path(__file__).resolve().parent
TEST_CASES_FILE = EVAL_DIR / "react_test_cases.json"
RESULTS_FILE = EVAL_DIR / "react_results.json"


JUDGE_PROMPT = """You are an evaluator scoring a travel itinerary produced by an AI agent.

Score the response on each of the following dimensions from 1 (poor) to 5 (excellent):
- relevance: Does the response address the user's specific request and constraints?
- completeness: Does it cover transportation, accommodation, daily activities, dining, and budget?
- personalization: Does it reflect the group type, travel style, and any special requests?
- practicality: Are the recommendations realistic with concrete names, links, and costs?

Reply with ONLY a single JSON object in this exact format:
{"relevance": <int>, "completeness": <int>, "personalization": <int>, "practicality": <int>, "comment": "<one short sentence>"}
"""


def llm_judge(user_message: str, agent_response: str) -> dict:
    """Use the LLM to score the agent output."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"USER REQUEST:\n{user_message}\n\nAGENT RESPONSE:\n{agent_response}"},
            ],
            max_tokens=300,
        )
        text = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        return {"error": str(e)}
    return {"error": "could not parse judge response"}


def extract_total_cost(response: str) -> float | None:
    """Try to find a total dollar amount in the agent's budget summary."""
    if not response:
        return None
    candidates = re.findall(r"total[^$]{0,40}\$([0-9,]+(?:\.[0-9]{1,2})?)", response, re.IGNORECASE)
    if not candidates:
        return None
    values = []
    for c in candidates:
        try:
            values.append(float(c.replace(",", "")))
        except ValueError:
            pass
    if not values:
        return None
    return max(values)


def evaluate_case(case: dict) -> dict:
    """Run a single test case and compute metrics."""
    print(f"\n=== Running: {case['name']} ===")
    start = time.time()
    try:
        result = run_agent(case["user_message"], SYSTEM_PROMPT)
    except Exception as e:
        return {
            "id": case["id"],
            "name": case["name"],
            "error": str(e),
            "completed": False,
        }
    elapsed = time.time() - start

    response_text = result.get("response", "")
    tool_calls = result.get("tool_calls", [])
    tools_used = [c["tool"] for c in tool_calls]

    completed = bool(response_text.strip()) and result.get("iterations", 0) < 15

    expected_tools = set(case.get("expected_tools", []))
    forbidden_tools = set(case.get("forbidden_tools", []))
    used_set = set(tools_used)
    expected_present = len(expected_tools & used_set) / max(len(expected_tools), 1)
    forbidden_violation = len(forbidden_tools & used_set) > 0

    budget = case.get("budget")
    total_cost = extract_total_cost(response_text)
    if budget is None:
        budget_ok = None
    elif total_cost is None:
        budget_ok = None
    else:
        budget_ok = total_cost <= float(budget) * 1.10  # 10% tolerance

    judge_scores = llm_judge(case["user_message"], response_text)

    return {
        "id": case["id"],
        "name": case["name"],
        "completed": completed,
        "iterations": result.get("iterations", 0),
        "num_tool_calls": len(tool_calls),
        "tools_used": tools_used,
        "expected_tool_coverage": round(expected_present, 2),
        "forbidden_tool_violation": forbidden_violation,
        "budget": budget,
        "estimated_total_cost": total_cost,
        "budget_ok": budget_ok,
        "elapsed_seconds": round(elapsed, 1),
        "judge_scores": judge_scores,
        "response_preview": response_text[:500],
    }


def summarize(results: list[dict]) -> dict:
    """Compute aggregate metrics across all results."""
    n = len(results)
    completed = sum(1 for r in results if r.get("completed"))
    avg_iters = sum(r.get("iterations", 0) for r in results) / max(n, 1)
    avg_tool_calls = sum(r.get("num_tool_calls", 0) for r in results) / max(n, 1)
    avg_coverage = sum(r.get("expected_tool_coverage", 0) for r in results) / max(n, 1)
    forbidden_violations = sum(1 for r in results if r.get("forbidden_tool_violation"))

    budget_evaluated = [r for r in results if r.get("budget_ok") is not None]
    budget_pass_rate = (
        sum(1 for r in budget_evaluated if r["budget_ok"]) / len(budget_evaluated)
        if budget_evaluated else None
    )

    judge_dims = ["relevance", "completeness", "personalization", "practicality"]
    judge_avgs = {}
    for dim in judge_dims:
        scores = [r["judge_scores"].get(dim) for r in results
                  if isinstance(r.get("judge_scores"), dict) and isinstance(r["judge_scores"].get(dim), (int, float))]
        judge_avgs[dim] = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "num_cases": n,
        "task_completion_rate": round(completed / max(n, 1), 2),
        "avg_iterations": round(avg_iters, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "avg_expected_tool_coverage": round(avg_coverage, 2),
        "forbidden_tool_violations": forbidden_violations,
        "budget_pass_rate": round(budget_pass_rate, 2) if budget_pass_rate is not None else None,
        "judge_averages": judge_avgs,
    }


def _judge_avg(scores: dict) -> float | None:
    if not isinstance(scores, dict):
        return None
    dims = ["relevance", "completeness", "personalization", "practicality"]
    vals = [scores.get(d) for d in dims if isinstance(scores.get(d), (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def print_results_table(results: list[dict]):
    """Print a per-case summary table to the terminal."""
    headers = ["ID", "OK", "Iter", "Tools", "ToolAcc", "Forbid", "Budget", "Quality", "Time(s)"]
    widths = [22, 3, 5, 6, 8, 7, 12, 8, 8]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, widths))

    print("\n" + fmt_row(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in results:
        if r.get("error"):
            print(fmt_row([r["id"], "✗", "-", "-", "-", "-", "-", "-", "-"]))
            continue
        budget_cell = "n/a" if r.get("budget_ok") is None else ("ok" if r["budget_ok"] else "over")
        if r.get("estimated_total_cost") is not None and r.get("budget"):
            budget_cell = f"${int(r['estimated_total_cost'])}/${r['budget']}"
        quality = _judge_avg(r.get("judge_scores", {}))
        print(fmt_row([
            r["id"],
            "✓" if r.get("completed") else "✗",
            r.get("iterations", "-"),
            r.get("num_tool_calls", "-"),
            r.get("expected_tool_coverage", "-"),
            "yes" if r.get("forbidden_tool_violation") else "no",
            budget_cell,
            quality if quality is not None else "-",
            r.get("elapsed_seconds", "-"),
        ]))


def main():
    with open(TEST_CASES_FILE) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        results.append(evaluate_case(case))

    summary = summarize(results)

    output = {"summary": summary, "results": results}
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print_results_table(results)

    print("\n\n=========== AGGREGATE SUMMARY ===========")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
