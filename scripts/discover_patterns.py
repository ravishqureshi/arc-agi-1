#!/usr/bin/env python3
"""
Empirical Pattern Discovery for ARC Tasks

Sample failing tasks and analyze them to discover:
- Tiling/repetition patterns
- Drawing patterns (lines, boxes)
- Other common transformations

Goal: Learn what operators we need from actual failing tasks.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_solver import G, solve_instance, ARCInstance

def load_tasks():
    """Load training tasks."""
    data_dir = Path(__file__).parent.parent / 'data'

    with open(data_dir / 'arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open(data_dir / 'arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    return challenges, solutions

def detect_tiling(input_grid, output_grid):
    """
    Detect if output is a tiled version of input or a subgrid.
    Returns: (is_tiling, description)
    """
    h_in, w_in = input_grid.shape
    h_out, w_out = output_grid.shape

    # Check if output is exact repetition of input
    if h_out % h_in == 0 and w_out % w_in == 0:
        nx = h_out // h_in
        ny = w_out // w_in
        if nx > 1 or ny > 1:
            tiled = np.tile(input_grid, (nx, ny))
            if np.array_equal(tiled, output_grid):
                return True, f"Tile input {nx}x{ny} times"

    # Check if input is a tile that needs to be repeated
    # (find smallest repeating pattern in input)
    for h_pat in range(1, h_in // 2 + 1):
        if h_out % h_pat == 0:
            for w_pat in range(1, w_in // 2 + 1):
                if w_out % w_pat == 0:
                    pattern = input_grid[:h_pat, :w_pat]
                    nx = h_out // h_pat
                    ny = w_out // w_pat
                    tiled = np.tile(pattern, (nx, ny))
                    if np.array_equal(tiled, output_grid):
                        return True, f"Tile {h_pat}x{w_pat} pattern {nx}x{ny} times"

    return False, None

def detect_drawing(input_grid, output_grid):
    """
    Detect if output has lines or boxes drawn on input.
    Returns: (has_drawing, description)
    """
    if input_grid.shape != output_grid.shape:
        return False, None

    diff = (input_grid != output_grid)
    if not np.any(diff):
        return False, None

    changed_pixels = np.argwhere(diff)
    if len(changed_pixels) == 0:
        return False, None

    rows = changed_pixels[:, 0]
    cols = changed_pixels[:, 1]

    # Check for horizontal line
    if len(np.unique(rows)) == 1:
        return True, f"Horizontal line at row {rows[0]}"

    # Check for vertical line
    if len(np.unique(cols)) == 1:
        return True, f"Vertical line at col {cols[0]}"

    # Check for box (hollow rectangle)
    if len(np.unique(rows)) == 2 or len(np.unique(cols)) == 2:
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # Count boundary vs interior
        boundary_count = 0
        for r, c in changed_pixels:
            if r in [r_min, r_max] or c in [c_min, c_max]:
                boundary_count += 1

        if boundary_count / len(changed_pixels) > 0.7:
            return True, f"Box/rectangle at ({r_min},{c_min})-({r_max},{c_max})"

    return False, None

def detect_shift(input_grid, output_grid):
    """Detect if output is shifted version of input."""
    if input_grid.shape != output_grid.shape:
        return False, None

    h, w = input_grid.shape

    # Try different shifts
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue

            # Create shifted version
            shifted = np.zeros_like(input_grid)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        shifted[nr, nc] = input_grid[r, c]

            if np.array_equal(shifted, output_grid):
                return True, f"Shift by ({dr},{dc})"

    return False, None

def analyze_task(task_id, task_data, solutions):
    """Analyze a single task for patterns."""
    patterns = {
        'tiling': False,
        'drawing': False,
        'shift': False,
        'other': False
    }
    details = []

    try:
        train_pairs = task_data['train']

        for i, pair in enumerate(train_pairs):
            inp = np.array(pair['input'], dtype=int)
            out = np.array(pair['output'], dtype=int)

            # Check tiling
            is_tiling, desc = detect_tiling(inp, out)
            if is_tiling:
                patterns['tiling'] = True
                details.append(f"Train {i}: {desc}")

            # Check drawing
            has_drawing, desc = detect_drawing(inp, out)
            if has_drawing:
                patterns['drawing'] = True
                details.append(f"Train {i}: {desc}")

            # Check shift
            is_shift, desc = detect_shift(inp, out)
            if is_shift:
                patterns['shift'] = True
                details.append(f"Train {i}: {desc}")

        if not any(patterns.values()):
            patterns['other'] = True

    except Exception as e:
        details.append(f"Error: {e}")

    return patterns, details

def main():
    print("=== ARC Pattern Discovery ===\n")
    print("Loading tasks...")
    challenges, solutions = load_tasks()

    # Sample failing tasks (first 100)
    task_list = list(challenges.items())[:100]

    print(f"Analyzing {len(task_list)} tasks...\n")

    pattern_counts = defaultdict(int)
    examples = defaultdict(list)

    for task_id, task_data in task_list:
        patterns, details = analyze_task(task_id, task_data, solutions)

        for pattern_type, found in patterns.items():
            if found:
                pattern_counts[pattern_type] += 1
                if len(examples[pattern_type]) < 5:  # Keep first 5 examples
                    examples[pattern_type].append((task_id, details))

    # Report
    print("=" * 70)
    print("PATTERN DISCOVERY RESULTS")
    print("=" * 70)
    print(f"\nSample size: {len(task_list)} tasks\n")

    for pattern_type in ['tiling', 'drawing', 'shift', 'other']:
        count = pattern_counts[pattern_type]
        pct = 100 * count / len(task_list)
        print(f"{pattern_type.upper()}: {count}/{len(task_list)} ({pct:.1f}%)")

        if examples[pattern_type]:
            print(f"  Examples:")
            for task_id, details in examples[pattern_type][:3]:
                print(f"    {task_id}: {details[0] if details else 'N/A'}")
        print()

    print("=" * 70)
    print("\nNext steps:")
    print("1. Manually inspect example tasks")
    print("2. Design operators based on patterns")
    print("3. Implement + test on these tasks")

if __name__ == "__main__":
    main()
