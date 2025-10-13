#!/usr/bin/env python3
"""
Example: Track ARC-1 vs ARC-2 Performance Separately
=====================================================

This demonstrates how to use arc_version_split.py to compute
separate accuracy scores for ARC-AGI-1 vs ARC-AGI-2 tasks.
"""

import json
from arc_version_split import is_arc1_task, is_arc2_new_task, get_version_stats

def compute_separate_scores(predictions_dict, solutions_dict):
    """
    Compute accuracy separately for ARC-1 vs ARC-2 tasks.

    Args:
        predictions_dict: {task_id: [output_grid_1, output_grid_2, ...]}
        solutions_dict: {task_id: [expected_grid_1, expected_grid_2, ...]}

    Returns:
        Dictionary with separate scores for ARC-1 and ARC-2
    """
    arc1_correct = 0
    arc1_total = 0
    arc2_correct = 0
    arc2_total = 0

    for task_id, predicted_outputs in predictions_dict.items():
        if task_id not in solutions_dict:
            continue

        expected_outputs = solutions_dict[task_id]

        # Check if all test outputs match
        task_correct = (predicted_outputs == expected_outputs)

        # Count by version
        if is_arc1_task(task_id):
            arc1_total += 1
            if task_correct:
                arc1_correct += 1
        elif is_arc2_new_task(task_id):
            arc2_total += 1
            if task_correct:
                arc2_correct += 1

    return {
        'arc1_score': arc1_correct / arc1_total if arc1_total > 0 else 0,
        'arc1_correct': arc1_correct,
        'arc1_total': arc1_total,
        'arc2_score': arc2_correct / arc2_total if arc2_total > 0 else 0,
        'arc2_correct': arc2_correct,
        'arc2_total': arc2_total,
        'overall_score': (arc1_correct + arc2_correct) / (arc1_total + arc2_total)
                         if (arc1_total + arc2_total) > 0 else 0,
        'overall_correct': arc1_correct + arc2_correct,
        'overall_total': arc1_total + arc2_total
    }


if __name__ == '__main__':
    # Print dataset stats
    print('Dataset Composition')
    print('=' * 60)
    stats = get_version_stats()
    for key, val in stats.items():
        print(f'{key}: {val}')

    print('\n' + '=' * 60)
    print('Example: How to track separate performance')
    print('=' * 60)

    # Load training challenges
    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    # Example: Check first 5 tasks
    print('\nFirst 5 training tasks:')
    for i, task_id in enumerate(list(challenges.keys())[:5]):
        version = 'ARC-1' if is_arc1_task(task_id) else 'ARC-2 (new)'
        print(f'  {i+1}. {task_id} → {version}')

    print('\nRemoved tasks from ARC-1 (not in ARC-2):')
    print('  0dfd9992, 29ec7d0e, 3631a71a, 40853293, 73251a56,')
    print('  9ecd008a, a3df8b1e, c3f564a4, dc0a314f')

    print('\n' + '=' * 60)
    print('Usage in your solver:')
    print('=' * 60)
    print("""
# After getting predictions and solutions:
results = compute_separate_scores(predictions, solutions)

print(f"ARC-1 Score: {results['arc1_score']:.1%} ({results['arc1_correct']}/{results['arc1_total']})")
print(f"ARC-2 Score: {results['arc2_score']:.1%} ({results['arc2_correct']}/{results['arc2_total']})")
print(f"Overall:     {results['overall_score']:.1%} ({results['overall_correct']}/{results['overall_total']})")
    """)
