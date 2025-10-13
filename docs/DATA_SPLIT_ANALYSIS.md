# ARC-AGI Dataset Split Analysis

**Date:** 2025-10-13
**Critical Finding:** Our current implementation has 0% accuracy on unseen ARC-2 tasks.

---

## Executive Summary

We have proper train/eval/test splits, but we discovered a **critical performance gap**:

| Dataset | Tasks | Test Outputs | ARC-1 | ARC-2 | Our Performance |
|---------|-------|--------------|-------|-------|-----------------|
| **Training** | 1,000 | 1,076 | 391 (39%) | 609 (61%) | **1.3%** (14/1076) |
| **Evaluation** | 120 | 172 | 0 (0%) | 120 (100%) | **0.0%** (0/172) ⚠️ |
| **Test (Hidden)** | 240 | ~320 | ??? | Mostly ARC-2 | **Expected: ~0%** ⚠️ |

**Key Insight:** All 14 passing tasks are ARC-1. We solve 0/609 ARC-2 training tasks and 0/120 ARC-2 evaluation tasks.

---

## Dataset Structure

### Training Set (`arc-agi_training_challenges.json`)
- **Purpose:** Develop and tune algorithms
- **Size:** 1,000 tasks (1,076 test outputs)
- **Composition:**
  - ARC-1 (original 2019): 391 tasks (39.1%)
  - ARC-2 (new 2024): 609 tasks (60.9%)
- **Has solutions:** Yes ✅

### Evaluation Set (`arc-agi_evaluation_challenges.json`)
- **Purpose:** Validate generalization to UNSEEN tasks
- **Size:** 120 tasks (172 test outputs)
- **Composition:**
  - ARC-1: 0 tasks (0%)
  - ARC-2: 120 tasks (100%) ⚠️
- **Has solutions:** Yes ✅
- **Status:** NOT USED until now

### Test Set (`arc-agi_test_challenges.json`)
- **Purpose:** Final Kaggle submission benchmark
- **Size:** 240 tasks (~320 test outputs)
- **Composition:** Unknown (likely 100% ARC-2)
- **Has solutions:** No ❌ (hidden on Kaggle)
- **Note:** Local file is placeholder; real test set loaded during Kaggle submission

---

## Two Types of Data Splits

### 1. Per-Task Train/Test Split (✅ We do this correctly)

Each task contains:
```json
{
  "task_id": {
    "train": [
      {"input": [[grid]], "output": [[grid]]}  // Learn from these
    ],
    "test": [
      {"input": [[grid]]}  // Predict these (output held out)
    ]
  }
}
```

**What we do:**
1. Induce rules from `train` pairs only
2. Apply rules to predict `test` outputs
3. Compare predictions to ground truth (during evaluation)

**Result:** No per-task data leakage ✅

### 2. Dataset-Level Train/Eval/Test Split (❌ We were NOT doing)

```
Develop on TRAINING set → Validate on EVALUATION set → Submit on TEST set
```

**What we SHOULD do:**
1. **Training phase:** Develop operators using training set (1000 tasks)
2. **Validation phase:** Test on evaluation set (120 UNSEEN tasks)
3. **Submission phase:** Submit to Kaggle with test set (240 hidden tasks)

**What we WERE doing:**
- Only testing on training set (implicit overfitting risk)
- Never measured generalization to new tasks

---

## Performance Breakdown

### Training Set Results (1,076 test outputs)

| Category | Passing | Total | Accuracy |
|----------|---------|-------|----------|
| **ARC-1 tasks** | 14 | 407 | **3.4%** ✅ |
| **ARC-2 tasks** | 0 | 669 | **0.0%** ❌ |
| **Overall** | 14 | 1,076 | **1.3%** |

**Rules that work:**
- ROT (3), COLOR_PERM (3), MOVE_OBJ_RANK (2), FLIP (2)
- RECOLOR_OBJ_RANK (1), CROP_BBOX_NONZERO (1)
- BORDER_PAINT (1), TILE (1)

**All 14 passing tasks are ARC-1!**

### Evaluation Set Results (172 test outputs)

| Category | Passing | Total | Accuracy |
|----------|---------|-------|----------|
| **ARC-2 tasks** | 0 | 172 | **0.0%** ❌ |

**Rules that work:** None

**Finding:** Our rules DO NOT generalize to unseen ARC-2 tasks.

### Expected Competition Performance

Based on evaluation results:
- **Training:** 1.3% (misleading - mostly ARC-1 successes)
- **Evaluation:** 0.0% (honest estimate)
- **Test (Kaggle):** ~0% (expected to match evaluation)

---

## Why ARC-2 is Different

### ARC-1 Tasks (2019)
- Simple spatial transformations (rotate, flip, crop)
- Direct color mappings
- Object operations (move, copy, recolor by size)
- Pattern: **explicit rules**

### ARC-2 Tasks (2024)
- Symbol learning (objects represent abstract concepts)
- In-context definition (meanings defined within task)
- Multi-step compositions
- Relational reasoning
- Pattern: **implicit abstractions**

### What Our Rules Can Do
- ✅ Symmetry (ROT, FLIP)
- ✅ Simple color permutations
- ✅ Crop to bounding box
- ✅ Object rank operations (move/copy/recolor largest)
- ✅ Tiling
- ✅ Border painting

### What Our Rules CANNOT Do
- ❌ Learn symbols (e.g., "this shape means 'grow'")
- ❌ Abstract reasoning (e.g., "apply rule from train pairs")
- ❌ Complex compositions (e.g., "detect, transform, place")
- ❌ Relational operations (e.g., "connect nearest objects")
- ❌ Physics simulation (e.g., "apply gravity")
- ❌ Path finding (e.g., "trace shortest path")

---

## Recommended Workflow

### ✅ Phase 0: Baseline (COMPLETED)
- **Goal:** Restore legacy demo rules
- **Dataset:** Training set (1000 tasks)
- **Results:** 14/1076 = 1.3%
- **Status:** ✅ EXCEEDED legacy (12 → 14 tasks)

### 🔄 Phase 1: Evaluation-Driven Development (CURRENT)
- **Goal:** Identify and fix ARC-2 gaps
- **Dataset:** Use evaluation set for guidance
- **Workflow:**
  1. Analyze evaluation failures
  2. Identify missing operators
  3. Implement operators on training set
  4. Re-test on evaluation set
  5. Iterate until 60-80% evaluation accuracy
- **Status:** 🔄 Just started (0/172 = 0%)

### 🎯 Phase 2: Scale to Competition Target
- **Goal:** Reach 60-80% on evaluation set
- **Dataset:** Train on training, validate on evaluation
- **Expected:** 103-137 evaluation tasks passing
- **Operators needed:** 40-60 additional operators

### 🏆 Phase 3: Kaggle Submission
- **Goal:** Submit to competition
- **Dataset:** Hidden test set (240 tasks)
- **Expected:** Match evaluation performance (~60-80%)

---

## Data Leakage Analysis

### Is There Data Leakage?

**Per-Task:** ✅ No leakage
- Rules induced from train pairs only
- Test outputs held out during learning
- Predictions made without seeing test outputs

**Dataset-Level:** ⚠️ Implicit tuning risk
- We develop operators that work on training tasks
- We tune catalog ordering for training task patterns
- Risk: Overfitting to training task types

**Mitigation:**
- Use evaluation set to measure true generalization
- Only use training set for development
- Never tune operators specifically for evaluation set

---

## How to Use This Analysis

### For Development
```bash
# Develop on training set (iterate freely)
python scripts/generate_test_coverage.py

# Validate on evaluation set (measure generalization)
python scripts/test_on_evaluation.py
```

### For Reporting
- **Training accuracy:** Shows what we can solve
- **Evaluation accuracy:** Shows what we WILL solve on Kaggle
- **Gap:** Shows how much overfitting/brittleness we have

### For Planning
1. Focus on evaluation failures
2. Implement operators for ARC-2 patterns
3. Re-test on evaluation to measure progress
4. Target: 60-80% evaluation accuracy before submission

---

## Files

- **Training data:** `data/arc-agi_training_challenges.json` (1000 tasks)
- **Training solutions:** `data/arc-agi_training_solutions.json`
- **Evaluation data:** `data/arc-agi_evaluation_challenges.json` (120 tasks)
- **Evaluation solutions:** `data/arc-agi_evaluation_solutions.json`
- **Test data:** `data/arc-agi_test_challenges.json` (240 tasks, placeholder)
- **Training results:** `docs/test_coverage_data.json`
- **Evaluation results:** `docs/evaluation_results.json`

---

## Next Steps

1. ✅ Document data split structure (this file)
2. ✅ Run evaluation to measure true performance (0/172)
3. 🔄 Analyze evaluation failures to identify missing operators
4. 🔄 Implement ARC-2-specific operators (symbol learning, composition)
5. 🔄 Iterate: develop on training, validate on evaluation
6. 🎯 Target: 60-80% on evaluation set
7. 🏆 Submit to Kaggle when evaluation target reached

---

**Key Takeaway:** We have proper data splits, but our 1.3% training accuracy is misleading. True generalization to ARC-2 tasks is 0%. We need 40-60 new operators focused on ARC-2 patterns to be competition-ready.
