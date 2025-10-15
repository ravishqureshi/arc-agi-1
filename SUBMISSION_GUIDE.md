# ARC-AGI Submission Guide
## B0+B1 Fixed-Point Closure Solver

**Date:** 2025-10-15
**Status:** READY FOR SUBMISSION
**Review:** See `reviews/submission_determinism_agent_review.md`

---

## Quick Start (5 Steps)

### 1. Verify Determinism

```bash
# Set deterministic seed
export PYTHONHASHSEED=0

# Run B1 determinism tests (should show 4/4 passed)
PYTHONPATH=src python scripts/verify_b1_determinism.py

# Run pipeline tests (should show 3/3 passed)
python scripts/test_pipeline.py
```

**Expected Output:**
```
✓ B1 implementation is DETERMINISTIC and PARAMETRIC
✓ Pipeline is DETERMINISTIC and SCHEMA-COMPLIANT
```

---

### 2. Generate Predictions

```bash
# Set deterministic seed
export PYTHONHASHSEED=0

# Run solver on evaluation set (400 tasks)
python scripts/run_public.py \
    --dataset=data/arc-agi_evaluation_challenges.json \
    --output=runs/submission

# This will create:
#   runs/submission/predictions.json
#   runs/submission/receipts.jsonl
```

**Runtime:** ~1-5 minutes (B1 only, single closure family)

---

### 3. Validate Output

```bash
# Validate predictions.json schema
python scripts/submission_validator.py runs/submission/predictions.json

# Expected output:
# ✓ Validated 400 tasks
# ✓ All grids are valid 2D lists of integers 0-9
# ✓ Schema is compliant with Kaggle ARC-AGI submission format
```

---

### 4. Verify Determinism (Full Test)

```bash
# Run determinism check (runs solver twice, compares byte-for-byte)
bash scripts/determinism.sh data/arc-agi_evaluation_challenges.json

# Expected output:
# ✓ predictions.json: IDENTICAL
# ✓ receipts.jsonl: IDENTICAL
# ✓ DETERMINISM CHECK PASSED
```

**Runtime:** ~2-10 minutes (runs solver twice)

---

### 5. Create Submission Zip

```bash
# Create submission zip (only predictions.json required)
cd runs/submission
zip submission.zip predictions.json
cd ../..

# Verify zip contents
unzip -l runs/submission/submission.zip
```

**Expected Output:**
```
Archive:  submission.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
   XXXXXX  2025-10-15 XX:XX   predictions.json
---------                     -------
   XXXXXX                     1 file
```

---

### 6. Submit to Kaggle

1. Go to: https://www.kaggle.com/competitions/arc-prize-2024
2. Click "Submit Predictions"
3. Upload: `runs/submission/submission.zip`
4. Add description: "B0+B1 Fixed-Point Closure Solver (KEEP_LARGEST_COMPONENT)"
5. Click "Submit"

---

## Expected Coverage (B1 Only)

**Estimated Tasks Solved:** 5-15 out of 400 (~1-4%)

**Rationale:**
- B1 implements only KEEP_LARGEST_COMPONENT closure
- This is a narrow pattern in ARC-AGI
- Used as baseline to verify submission pipeline

**Next Steps for Higher Coverage:**
- Implement B2: OUTLINE_OBJECTS → +10-20 tasks
- Implement B3: MOD_PATTERN → +20-30 tasks
- Implement B4: SYMMETRY_COMPLETION → +5-10 tasks

---

## Troubleshooting

### Issue: Validator fails with "Value must be int"

**Cause:** Numpy arrays not converted to lists

**Fix:** Ensure `pred.tolist()` is called (see `run_public.py:82`)

---

### Issue: Determinism check fails

**Cause:** Non-deterministic code path or environment variable not set

**Fix:**
```bash
# Always set PYTHONHASHSEED before running
export PYTHONHASHSEED=0

# Check for dict iteration or set iteration in code
# See review: reviews/submission_determinism_agent_review.md Section 5
```

---

### Issue: predictions.json is empty or has missing tasks

**Cause:** Solver crashed or dataset path incorrect

**Fix:**
```bash
# Check dataset exists
ls -la data/arc-agi_evaluation_challenges.json

# Run with verbose output (no --quiet flag)
python scripts/run_public.py \
    --dataset=data/arc-agi_evaluation_challenges.json \
    --output=runs/debug
```

---

### Issue: Submission rejected by Kaggle

**Possible Causes:**
1. Task IDs missing `.json` extension
2. Predictions are numpy arrays (not lists)
3. Values outside 0-9 range
4. Grid rows have inconsistent lengths

**Fix:**
```bash
# Run validator to check schema
python scripts/submission_validator.py runs/submission/predictions.json

# If validator passes but Kaggle rejects, check:
# - Zip file only contains predictions.json (no other files)
# - JSON is not corrupted (validate with jq)
jq . runs/submission/predictions.json > /dev/null && echo "Valid JSON"
```

---

## File Locations

### Outputs
- `runs/submission/predictions.json` - Required for Kaggle submission
- `runs/submission/receipts.jsonl` - Debugging and analysis (not submitted)
- `runs/submission/submission.zip` - Upload to Kaggle

### Scripts
- `scripts/run_public.py` - Main entry point
- `scripts/submission_validator.py` - Schema validation
- `scripts/determinism.sh` - Determinism verification
- `scripts/verify_b1_determinism.py` - B1-specific tests
- `scripts/test_pipeline.py` - Pipeline integration tests

### Core Implementation
- `src/arc_solver/closure_engine.py` - Fixed-point engine (B0)
- `src/arc_solver/closures.py` - Closure implementations (B1)
- `src/arc_solver/search.py` - Orchestration
- `src/arc_solver/utils.py` - Utilities and hashing

### Documentation
- `reviews/submission_determinism_agent_review.md` - Full determinism audit
- `docs/IMPLEMENTATION_PLAN_v2.md` - Implementation plan
- `docs/CONTEXT_INDEX.md` - Repository navigation

---

## Determinism Guarantees

### Sources of Determinism:
1. **PYTHONHASHSEED=0** - Consistent hash functions
2. **Fixed iteration order** - Closures applied in sequence
3. **Lexicographic tiebreaking** - Largest component selection (size, -bbox[0], -bbox[1])
4. **Lowest-color fallback** - Multi-valued cells pick lowest color
5. **Row-major scan** - BFS component discovery
6. **Sorted keys** - JSON hashing with sort_keys=True

### Verified Through:
- B1 unit tests (4/4 passed)
- Pipeline tests (3/3 passed)
- Byte-level hash comparison (determinism.sh)

**See:** `reviews/submission_determinism_agent_review.md` Section 5 for complete analysis

---

## Contact & Support

**Review Document:** `reviews/submission_determinism_agent_review.md`

**Key Findings:**
- ✓ VERDICT: PASS (deterministic and submission-ready)
- ✓ BLOCKERS: None
- ⚠ WARNINGS: 2 medium severity (dict iteration, status tracking)
- ✓ SCHEMA: Compliant with Kaggle format

**Next Steps:**
1. Run submission (follow 5-step guide above)
2. Implement B2-B9 closure families for higher coverage
3. Monitor Kaggle leaderboard

---

**Good luck!**
