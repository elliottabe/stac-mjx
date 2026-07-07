#!/bin/bash
# Example script for running STAC multirun across dataset versions
# This demonstrates how to process multiple versions in one command

set -e  # Exit on error

echo "=========================================="
echo "STAC MULTIRUN EXAMPLE"
echo "=========================================="
echo ""

# Define workspace paths
STAC_DIR="/home/eabe/Research/MyRepos/stac-mjx"
PREPROCESS_DIR="/home/eabe/Research/MyRepos/3d_tracking_dataset"

# Define versions to process
VERSIONS=(
    "Predictions_3D_20260114-145343"
    "Predictions_3D_20260202-171900"
    "Predictions_3D_20260203-103416"
)

echo "Processing ${#VERSIONS[@]} versions:"
for version in "${VERSIONS[@]}"; do
    echo "  - $version"
done
echo ""

# Step 1: Check if preprocessing is done (optional)
echo "Step 1: Checking preprocessing..."
for version in "${VERSIONS[@]}"; do
    preproc_file="/data2/users/eabe/datasets/Johnson_lab//${version}/preprocessed_bout.h5"
    if [ -f "$preproc_file" ]; then
        echo "  ✓ Found: $preproc_file"
    else
        echo "  ✗ Missing: $preproc_file"
        echo "    Run: cd $PREPROCESS_DIR && python scripts/preprocess_keypoints_for_ik.py paths=workstation dataset= version=$version"
    fi
done
echo ""

# Step 2: Test configuration (optional)
echo "Step 2: Testing configuration for first version..."
cd "$STAC_DIR"
python test_stac_configs.py \
    paths=workstation \
    dataset= \
    version="${VERSIONS[0]}"
echo ""

# Step 3: Run STAC multirun
echo "Step 3: Running STAC multirun..."
echo "Command:"
VERSION_LIST=$(IFS=,; echo "${VERSIONS[*]}")
echo "  python demos/run_stac_fly_model.py -m \\"
echo "    paths=workstation \\"
echo "    dataset= \\"
echo "    run_id=multirun_example \\"
echo "    version=$VERSION_LIST"
echo ""

# Uncomment to actually run
cd "$STAC_DIR/demos"
python run_stac_fly_model.py -m \
    paths=workstation \
    dataset= \
    run_id=multirun_example \
    version="$VERSION_LIST"

echo ""
echo "=========================================="
echo "MULTIRUN COMPLETE!"
echo "=========================================="
echo ""
echo "Output locations:"
for version in "${VERSIONS[@]}"; do
    output_dir="/data2/users/eabe/datasets/Johnson_lab//${version}/multirun_example"
    echo "  $output_dir"
done
echo ""
echo "Each directory contains:"
echo "  - Fruitfly_fit_V1_free.h5"
echo "  - Fruitfly_ik_V1_free.h5"
echo "  - _fruitfly_v1.mp4"
