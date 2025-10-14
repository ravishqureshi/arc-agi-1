#!/usr/bin/env python3
"""
Generate test coverage report as a clean table.

Table columns:
- Task ID
- Category (ARC-1 or ARC-2)
- Status (PASS/FAIL)
- Expected Output
- Computed Output
- Diff (residual)
- Operator Chain
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys

# Import our solver
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from ui_clean import (
    autobuild_operators, beam_search, ARCInstance, equal, residual, Grid, G
)

@dataclass
class TestResult:
    task_id: str
    category: str
    passed: bool
    train_pairs: int
    test_cases: int
    expected_shape: tuple
    computed_shape: Optional[tuple]
    expected_str: str
    computed_str: str
    residual_val: Optional[int]
    operator_chain: Optional[str]
    error: Optional[str]

def load_arc1_ids() -> set:
    """Load ARC-1 task IDs"""
    with open('arc1_task_ids.txt', 'r') as f:
        return set(line.strip() for line in f if line.strip())

def load_training_data() -> Tuple[Dict, Dict]:
    """Load training challenges and solutions"""
    with open('data/arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    with open('data/arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    return challenges, solutions

def grid_to_compact_str(arr: np.ndarray, max_size=30) -> str:
    """Convert grid to very compact string"""
    if arr.size <= max_size:
        # Show first row as sample
        first_row = arr[0].tolist() if arr.shape[0] > 0 else []
        return f"{arr.shape} {first_row[:5]}{'...' if len(first_row) > 5 else ''}"
    else:
        return f"{arr.shape}"

def solve_task(task_id: str, task_data: Dict, solutions: Dict, arc1_ids: set) -> TestResult:
    """Solve a single task and compare with expected solution"""
    category = "ARC-1" if task_id in arc1_ids else "ARC-2"

    try:
        # Convert train pairs to numpy arrays
        train = [(G(p['input']), G(p['output'])) for p in task_data['train']]

        # Get test inputs and expected outputs (use first test case)
        test_inputs = [G(t['input']) for t in task_data['test']]
        expected_outputs = [G(sol) for sol in solutions[task_id]]

        # Use first test case
        test_in = test_inputs[0]
        expected = expected_outputs[0]

        # Create instance and run solver
        inst = ARCInstance(
            name=task_id,
            train=train,
            test_in=test_inputs,
            test_out=expected_outputs
        )

        # Run beam search
        steps = beam_search(inst, max_depth=6, beam_size=160)

        if steps is None:
            return TestResult(
                task_id=task_id,
                category=category,
                passed=False,
                train_pairs=len(train),
                test_cases=len(test_inputs),
                expected_shape=expected.shape,
                computed_shape=None,
                expected_str=grid_to_compact_str(expected),
                computed_str="N/A",
                residual_val=None,
                operator_chain=None,
                error="No operators found"
            )

        # Verify on train
        train_pass = True
        for x, y in train:
            pred = x.copy()
            for op in steps:
                pred = op.prog(pred)
            if not equal(pred, y):
                train_pass = False
                break

        if not train_pass:
            return TestResult(
                task_id=task_id,
                category=category,
                passed=False,
                train_pairs=len(train),
                test_cases=len(test_inputs),
                expected_shape=expected.shape,
                computed_shape=None,
                expected_str=grid_to_compact_str(expected),
                computed_str="N/A",
                residual_val=None,
                operator_chain=" → ".join(op.name for op in steps),
                error="Train residual != 0"
            )

        # Apply to test
        computed = test_in.copy()
        for op in steps:
            computed = op.prog(computed)

        if computed.shape == expected.shape:
            res = residual(computed, expected)
            passed = equal(computed, expected)
        else:
            res = max(computed.size, expected.size)
            passed = False

        return TestResult(
            task_id=task_id,
            category=category,
            passed=passed,
            train_pairs=len(train),
            test_cases=len(test_inputs),
            expected_shape=expected.shape,
            computed_shape=computed.shape,
            expected_str=grid_to_compact_str(expected),
            computed_str=grid_to_compact_str(computed),
            residual_val=res,
            operator_chain=" → ".join(op.name for op in steps),
            error=None
        )

    except Exception as e:
        # Get expected output if possible
        expected_shape = None
        expected_str = "N/A"
        if task_id in solutions and len(solutions[task_id]) > 0:
            expected = G(solutions[task_id][0])
            expected_shape = expected.shape
            expected_str = grid_to_compact_str(expected)

        return TestResult(
            task_id=task_id,
            category=category,
            passed=False,
            train_pairs=len(task_data.get('train', [])),
            test_cases=len(task_data.get('test', [])),
            expected_shape=expected_shape,
            computed_shape=None,
            expected_str=expected_str,
            computed_str="N/A",
            residual_val=None,
            operator_chain=None,
            error=f"Exception: {str(e)[:50]}"
        )

def generate_table_report(results: List[TestResult], output_path: str):
    """Generate clean table report"""

    # Calculate statistics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    arc1_results = [r for r in results if r.category == "ARC-1"]
    arc2_results = [r for r in results if r.category == "ARC-2"]

    arc1_passed = sum(1 for r in arc1_results if r.passed)
    arc2_passed = sum(1 for r in arc2_results if r.passed)

    with open(output_path, 'w') as f:
        f.write("# ARC-AGI Test Coverage Report\n\n")
        f.write(f"**Generated by**: `generate_test_coverage.py`\n")
        f.write(f"**Solver**: `src/ui_clean.py` (8 inducers, beam search)\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Tasks**: {total}\n")
        f.write(f"- **Passed**: {passed} ({100*passed/total:.2f}%)\n")
        f.write(f"- **Failed**: {failed} ({100*failed/total:.2f}%)\n\n")

        f.write(f"### By Category\n\n")
        f.write(f"- **ARC-1**: {arc1_passed}/{len(arc1_results)} ({100*arc1_passed/len(arc1_results):.2f}%)\n")
        f.write(f"- **ARC-2**: {arc2_passed}/{len(arc2_results)} ({100*arc2_passed/len(arc2_results):.2f}%)\n\n")

        # Error breakdown
        no_ops = sum(1 for r in results if r.error and "No operators" in r.error)
        train_fail = sum(1 for r in results if r.error and "Train residual" in r.error)
        test_fail = sum(1 for r in results if not r.passed and r.error is None)

        f.write(f"### Failure Breakdown\n\n")
        f.write(f"- **No operators found**: {no_ops}\n")
        f.write(f"- **Train residual != 0**: {train_fail}\n")
        f.write(f"- **Test failed (ops found, train passed)**: {test_fail}\n\n")

        # Main table
        f.write("## Complete Test Results Table\n\n")
        f.write("| Task ID | Category | Status | Expected Shape | Computed Shape | Diff | Expected (sample) | Computed (sample) | Operator Chain | Error |\n")
        f.write("|---------|----------|--------|----------------|----------------|------|-------------------|-------------------|----------------|-------|\n")

        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            expected_shape = str(r.expected_shape) if r.expected_shape else "N/A"
            computed_shape = str(r.computed_shape) if r.computed_shape else "N/A"
            diff = str(r.residual_val) if r.residual_val is not None else "N/A"
            ops = r.operator_chain if r.operator_chain else "N/A"
            if len(ops) > 40:
                ops = ops[:37] + "..."
            err = r.error if r.error else "-"
            if len(err) > 30:
                err = err[:27] + "..."

            # Truncate strings for table
            expected_str = r.expected_str[:40] + "..." if len(r.expected_str) > 40 else r.expected_str
            computed_str = r.computed_str[:40] + "..." if len(r.computed_str) > 40 else r.computed_str

            f.write(f"| {r.task_id} | {r.category} | {status} | {expected_shape} | {computed_shape} | {diff} | {expected_str} | {computed_str} | {ops} | {err} |\n")

        f.write("\n")

        # Operator usage stats
        f.write("## Operator Usage Statistics\n\n")
        from collections import Counter
        op_counter = Counter()
        for r in results:
            if r.passed and r.operator_chain:
                # Get first operator in chain
                first_op = r.operator_chain.split(' → ')[0]
                op_counter[first_op] += 1

        if op_counter:
            f.write("| Operator | Tasks Solved |\n")
            f.write("|----------|-------------|\n")
            for op, count in op_counter.most_common():
                f.write(f"| {op} | {count} |\n")
        else:
            f.write("*No operators used successfully*\n")

def main():
    print("=" * 70)
    print("ARC-AGI Test Coverage Generator (Table Format)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    arc1_ids = load_arc1_ids()
    challenges, solutions = load_training_data()
    print(f"  - Loaded {len(challenges)} training tasks")
    print(f"  - ARC-1: {len(arc1_ids)} tasks")
    print(f"  - ARC-2: {len(challenges) - len(arc1_ids)} tasks")

    # Run solver
    print("\n[2/4] Running solver on all tasks...")
    results = []
    for idx, (task_id, task_data) in enumerate(challenges.items(), 1):
        if idx % 100 == 0:
            passed_so_far = sum(1 for r in results if r.passed)
            print(f"  Progress: {idx}/{len(challenges)} ({100*idx/len(challenges):.1f}%), Solved: {passed_so_far}")

        result = solve_task(task_id, task_data, solutions, arc1_ids)
        results.append(result)

        if result.passed:
            print(f"  ✓ {task_id}")

    # Generate report
    print("\n[3/4] Generating table report...")
    output_path = "docs/ARC_TEST_COVERAGE_REPORT.md"
    generate_table_report(results, output_path)
    print(f"  Report saved to: {output_path}")

    # Print summary
    print("\n[4/4] Summary:")
    passed = sum(1 for r in results if r.passed)
    arc1_results = [r for r in results if r.category == "ARC-1"]
    arc2_results = [r for r in results if r.category == "ARC-2"]
    arc1_passed = sum(1 for r in arc1_results if r.passed)
    arc2_passed = sum(1 for r in arc2_results if r.passed)

    print(f"  Total: {passed}/{len(results)} ({100*passed/len(results):.2f}%)")
    print(f"  ARC-1: {arc1_passed}/{len(arc1_results)} ({100*arc1_passed/len(arc1_results):.2f}%)")
    print(f"  ARC-2: {arc2_passed}/{len(arc2_results)} ({100*arc2_passed/len(arc2_results):.2f}%)")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
