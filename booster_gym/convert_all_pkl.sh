#!/bin/bash

# Batch PKL Conversion Script
# Converts all PKL files in the current directory for training compatibility
#
# Usage:
#   cd /path/to/your/pkl/files
#   bash /path/to/booster_gym/convert_all_pkl.sh
#
# Or from project root:
#   bash booster_gym/convert_all_pkl.sh

echo "🎭 Batch PKL Conversion for Motion Imitation Training"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER_SCRIPT="$SCRIPT_DIR/utils/convert_pkl_for_training.py"
OUTPUT_DIR="$SCRIPT_DIR/motion_data/t1_4dof"

# Check if converter script exists
if [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo "❌ Converter script not found: $CONVERTER_SCRIPT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count PKL files
PKL_COUNT=$(find . -maxdepth 1 -name "*.pkl" -type f | wc -l)

if [ $PKL_COUNT -eq 0 ]; then
    echo "ℹ️  No PKL files found in current directory: $(pwd)"
    echo "   Place your PKL files here and run this script again."
    exit 0
fi

echo "📁 Found $PKL_COUNT PKL file(s) in: $(pwd)"
echo "📤 Output directory: $OUTPUT_DIR"
echo ""

# Convert each PKL file
SUCCESS_COUNT=0
FAILED_COUNT=0

for pkl_file in *.pkl; do
    if [ -f "$pkl_file" ]; then
        echo "🔄 Converting: $pkl_file"
        output_file="$OUTPUT_DIR/$pkl_file"
        
        if python "$CONVERTER_SCRIPT" "$pkl_file" "$output_file"; then
            echo "✅ Success: $pkl_file -> $output_file"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ Failed: $pkl_file"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
        echo ""
    fi
done

echo "=================================================="
echo "📊 Conversion Summary:"
echo "   ✅ Successful: $SUCCESS_COUNT"
echo "   ❌ Failed: $FAILED_COUNT"
echo "   📁 Output: $OUTPUT_DIR"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "🎉 Converted files are ready for training!"
    echo "   To train: cd booster_gym && conda activate isaac-gym-training"
    echo "   Then run: python train_imitation.py --task T1Imitation"
fi