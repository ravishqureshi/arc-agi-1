#!/bin/bash
# Determinism verification script for ARC-AGI solver
#
# Runs solver twice with different configurations and verifies
# that outputs are byte-identical.
#
# Usage:
#   bash scripts/determinism.sh [dataset_path]
#
# Default dataset: data/arc-agi_evaluation_challenges.json

set -e

# Force deterministic Python behavior
export PYTHONHASHSEED=0

# Parse arguments
DATASET="${1:-data/arc-agi_evaluation_challenges.json}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="runs/determinism_${TIMESTAMP}"

echo "======================================================================"
echo "DETERMINISM CHECK"
echo "======================================================================"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "PYTHONHASHSEED: $PYTHONHASHSEED"
echo "======================================================================"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}/pass1"
mkdir -p "${OUTPUT_DIR}/pass2"

# Pass 1: Single-threaded (baseline)
echo "[1/2] Running Pass 1 (single-threaded baseline)..."
python scripts/run_public.py \
    --dataset="$DATASET" \
    --output="${OUTPUT_DIR}/pass1" \
    --quiet

echo "✓ Pass 1 complete"
echo ""

# Pass 2: Re-run with same configuration
echo "[2/2] Running Pass 2 (determinism check)..."
python scripts/run_public.py \
    --dataset="$DATASET" \
    --output="${OUTPUT_DIR}/pass2" \
    --quiet

echo "✓ Pass 2 complete"
echo ""

# Compare outputs
echo "======================================================================"
echo "COMPARING OUTPUTS"
echo "======================================================================"

# Compare predictions.json
echo "Comparing predictions.json..."
if diff -q "${OUTPUT_DIR}/pass1/predictions.json" "${OUTPUT_DIR}/pass2/predictions.json" > /dev/null; then
    echo "✓ predictions.json: IDENTICAL"
else
    echo "✗ predictions.json: DIFFERENT"
    echo ""
    echo "Showing differences:"
    diff "${OUTPUT_DIR}/pass1/predictions.json" "${OUTPUT_DIR}/pass2/predictions.json" || true
    echo ""
    echo "DETERMINISM CHECK FAILED: predictions.json differs between runs"
    exit 1
fi

# Compare receipts.jsonl
echo "Comparing receipts.jsonl..."
if diff -q "${OUTPUT_DIR}/pass1/receipts.jsonl" "${OUTPUT_DIR}/pass2/receipts.jsonl" > /dev/null; then
    echo "✓ receipts.jsonl: IDENTICAL"
else
    echo "✗ receipts.jsonl: DIFFERENT"
    echo ""
    echo "Showing differences:"
    diff "${OUTPUT_DIR}/pass1/receipts.jsonl" "${OUTPUT_DIR}/pass2/receipts.jsonl" || true
    echo ""
    echo "DETERMINISM CHECK FAILED: receipts.jsonl differs between runs"
    exit 1
fi

# Byte-level comparison
echo ""
echo "Byte-level comparison..."
HASH1_PRED=$(shasum -a 256 "${OUTPUT_DIR}/pass1/predictions.json" | awk '{print $1}')
HASH2_PRED=$(shasum -a 256 "${OUTPUT_DIR}/pass2/predictions.json" | awk '{print $1}')
HASH1_RECV=$(shasum -a 256 "${OUTPUT_DIR}/pass1/receipts.jsonl" | awk '{print $1}')
HASH2_RECV=$(shasum -a 256 "${OUTPUT_DIR}/pass2/receipts.jsonl" | awk '{print $1}')

echo "  predictions.json SHA-256:"
echo "    Pass 1: $HASH1_PRED"
echo "    Pass 2: $HASH2_PRED"
echo "  receipts.jsonl SHA-256:"
echo "    Pass 1: $HASH1_RECV"
echo "    Pass 2: $HASH2_RECV"

if [ "$HASH1_PRED" = "$HASH2_PRED" ] && [ "$HASH1_RECV" = "$HASH2_RECV" ]; then
    echo ""
    echo "======================================================================"
    echo "✓ DETERMINISM CHECK PASSED"
    echo "======================================================================"
    echo "Both runs produced byte-identical outputs."
    echo "The solver is deterministic and ready for submission."
    echo ""
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "✗ DETERMINISM CHECK FAILED"
    echo "======================================================================"
    echo "Byte-level hashes differ between runs."
    echo "This indicates non-deterministic behavior."
    echo ""
    exit 1
fi
