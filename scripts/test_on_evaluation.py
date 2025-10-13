#!/usr/bin/env python3
"""
Test ARC Solver on EVALUATION set (unseen tasks)
This measures true generalization to new tasks.
"""

import sys
sys.path.insert(0, 'src')

import json
from arc_solver import ARCInstance, solve_instance, G

def load_evaluation_tasks():
    """Load evaluation challenges and solutions."""
    with open('data/arc-agi_evaluation_challenges.json') as f:
        challenges = json.load(f)
    with open('data/arc-agi_evaluation_solutions.json') as f:
        solutions = json.load(f)
    return challenges, solutions

def test_evaluation_set():
    """Test on evaluation set and report results."""
    print("=" * 70)
    print("ARC Solver - EVALUATION SET (Unseen Tasks)")
    print("=" * 70)
    print()
    print("Loading evaluation tasks...")
    
    challenges, solutions = load_evaluation_tasks()
    
    print(f"Loaded {len(challenges)} tasks")
    print()
    
    # Track results
    total_tests = 0
    passing_tests = 0
    rule_counts = {}
    
    results = []
    
    print("Testing evaluation tasks...")
    task_count = 0
    
    for task_id, task_data in challenges.items():
        task_count += 1
        if task_count % 20 == 0:
            print(f"  Progress: {task_count}/{len(challenges)} tasks ({task_count/len(challenges)*100:.1f}%)")
        
        # Get train and test data
        train_pairs = [(G(p['input']), G(p['output'])) for p in task_data['train']]
        test_inputs = [G(p['input']) for p in task_data['test']]
        test_outputs = [G(out) for out in solutions[task_id]]
        
        # Test each test input
        for test_idx, (test_in, test_out) in enumerate(zip(test_inputs, test_outputs)):
            total_tests += 1
            
            # Create instance and solve
            inst = ARCInstance(
                name=f"{task_id}_test{test_idx}",
                train=train_pairs,
                test_in=[test_in],
                test_out=[test_out]
            )
            
            result = solve_instance(inst)
            
            # Check if correct
            match = result.acc_exact == 1.0 and result.receipts[0].residual == 0
            
            if match:
                passing_tests += 1
                rule_name = result.rule.name if result.rule else "None"
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
            
            results.append({
                'task_id': task_id,
                'test_idx': test_idx,
                'rule': result.rule.name if result.rule else "None",
                'residual': result.receipts[0].residual,
                'match': match
            })
    
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print()
    print(f"Total test outputs: {total_tests}")
    print(f"Passing: {passing_tests} ({passing_tests/total_tests*100:.2f}%)")
    print(f"Failing: {total_tests - passing_tests} ({(total_tests-passing_tests)/total_tests*100:.2f}%)")
    print()
    
    if rule_counts:
        print("Rules Used:")
        for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            print(f"  {rule}: {count}")
    else:
        print("⚠️  NO RULES MATCHED ANY EVALUATION TASKS!")
    
    print()
    print("=" * 70)
    print("COMPARISON: Training vs Evaluation")
    print("=" * 70)
    print()
    print(f"Training set:   14/1076 = 1.30% (ARC-1: 3.6%, ARC-2: 0.0%)")
    print(f"Evaluation set: {passing_tests}/172 = {passing_tests/172*100:.2f}% (ALL ARC-2)")
    print()
    
    if passing_tests < 5:
        print("⚠️  WARNING: Very low evaluation performance!")
        print("   This suggests our rules do NOT generalize to new ARC-2 tasks.")
        print("   We need more sophisticated operators for ARC-2 patterns.")
    
    print("=" * 70)
    
    # Save results
    output = {
        'total_tests': total_tests,
        'passing': passing_tests,
        'accuracy': passing_tests / total_tests,
        'rule_counts': rule_counts,
        'results': results
    }
    
    with open('docs/evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"Results saved to docs/evaluation_results.json")
    print()

if __name__ == '__main__':
    test_evaluation_set()
