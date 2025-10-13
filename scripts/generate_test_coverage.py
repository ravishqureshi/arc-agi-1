#!/usr/bin/env python3
"""
Generate test coverage report for ARC Solver.

Runs solver on all 1,000 training tasks, collects results, and updates:
- docs/test_coverage_data.json (raw data)
- docs/TEST_COVERAGE.md (human-readable report)
"""

import sys
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / 'src'))
sys.path.insert(0, str(root_dir))

from arc_solver import G, solve_instance, ARCInstance
from arc_version_split import is_arc1_task

def load_tasks():
    """Load all training tasks from data files."""
    data_dir = Path(__file__).parent.parent / 'data'

    with open(data_dir / 'arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open(data_dir / 'arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    return challenges, solutions

def run_full_test_sweep():
    """Run solver on all tasks and collect results."""
    print("Loading tasks...")
    challenges, solutions = load_tasks()

    results = []
    rule_counts = Counter()
    passing = 0
    arc1_pass = 0
    arc1_total = 0
    arc2_pass = 0
    arc2_total = 0

    total_tasks = len(challenges)
    print(f"Testing {total_tasks} tasks...")

    for task_idx, (task_id, task_data) in enumerate(challenges.items(), 1):
        if task_idx % 100 == 0:
            print(f"  Progress: {task_idx}/{total_tasks} tasks ({100*task_idx/total_tasks:.1f}%)")

        # Build ARCInstance
        train_pairs = [(G(pair['input']), G(pair['output'])) for pair in task_data['train']]
        test_inputs = [G(pair['input']) for pair in task_data['test']]
        test_outputs = [G(grid) for grid in solutions[task_id]]

        inst = ARCInstance(task_id, train_pairs, test_inputs, test_outputs)

        # Solve
        res = solve_instance(inst)

        # Determine ARC version
        arc_version = "ARC-1" if is_arc1_task(task_id) else "ARC-2"

        # Record results for each test output
        for test_idx in range(len(test_outputs)):
            pred = res.preds[test_idx]
            truth = test_outputs[test_idx]
            receipt = res.receipts[test_idx]

            match = receipt.residual == 0

            result = {
                "task_id": task_id,
                "arc_version": arc_version,
                "test_idx": test_idx,
                "rule": res.rule.name if res.rule else "None",
                "pred_shape": f"{pred.shape[0]}x{pred.shape[1]}",
                "truth_shape": f"{truth.shape[0]}x{truth.shape[1]}",
                "residual": receipt.residual,
                "match": match
            }

            results.append(result)

            if match:
                passing += 1
                if arc_version == "ARC-1":
                    arc1_pass += 1
                else:
                    arc2_pass += 1

                if res.rule:
                    rule_counts[res.rule.name] += 1

            if arc_version == "ARC-1":
                arc1_total += 1
            else:
                arc2_total += 1

    total_tests = len(results)
    accuracy = passing / total_tests if total_tests > 0 else 0

    # Count active rules (now includes MOVE/COPY)
    num_rules = 10  # Updated from 8 to 10

    metadata = {
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "solver_version": f"Modular v1.3.0 ({num_rules} active rules, MOVE/COPY added)",
        "total_tests": total_tests,
        "passing": passing,
        "accuracy": accuracy,
        "arc1_pass": arc1_pass,
        "arc1_total": arc1_total,
        "arc2_pass": arc2_pass,
        "arc2_total": arc2_total,
        "rule_counts": dict(rule_counts)
    }

    return metadata, results

def save_results(metadata, results):
    """Save results to JSON file."""
    output_file = Path(__file__).parent.parent / 'docs' / 'test_coverage_data.json'

    data = {
        "metadata": metadata,
        "all_results": results
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")

def print_summary(metadata):
    """Print summary of results."""
    print("\n" + "="*70)
    print("TEST COVERAGE SUMMARY")
    print("="*70)
    print(f"Generated: {metadata['generated']}")
    print(f"Solver: {metadata['solver_version']}")
    print(f"\nTotal Tests: {metadata['total_tests']}")
    print(f"Passing: {metadata['passing']} ({100*metadata['accuracy']:.2f}%)")
    print(f"ARC-1: {metadata['arc1_pass']}/{metadata['arc1_total']} ({100*metadata['arc1_pass']/metadata['arc1_total']:.2f}%)")
    print(f"ARC-2: {metadata['arc2_pass']}/{metadata['arc2_total']} ({100*metadata['arc2_pass']/metadata['arc2_total']:.2f}%)")
    print(f"\nRules Used:")
    for rule, count in sorted(metadata['rule_counts'].items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")
    print("="*70)

if __name__ == "__main__":
    print("ARC Solver - Test Coverage Generator")
    print("="*70)

    metadata, results = run_full_test_sweep()
    save_results(metadata, results)
    print_summary(metadata)

    print("\nNext: Update TEST_COVERAGE.md manually or run report generator")
