# PKL Conversion for Motion Imitation Training

This guide explains how to convert PKL files between environments for motion imitation training.

## The Problem

- **General Motion Retargeting (GMR)**: Runs only in Python 3.10
- **Isaac Gym Training**: Runs only in Python 3.8  
- **PKL files**: Generated in Python 3.10 may have numpy compatibility issues in Python 3.8

## The Solution

Use the conversion utility to make PKL files compatible across environments.

## Usage

### Single File Conversion

```bash
# From project root directory
conda activate general-motion-retargeting
python booster_gym/utils/convert_pkl_for_training.py input.pkl booster_gym/motion_data/t1_4dof/output.pkl
```

### Examples

```bash
# Convert a new dancing motion
python booster_gym/utils/convert_pkl_for_training.py new_dance.pkl booster_gym/motion_data/t1_4dof/new_dance.pkl

# Convert with verification
python booster_gym/utils/convert_pkl_for_training.py my_motion.pkl booster_gym/motion_data/t1_4dof/my_motion.pkl --verify
```

### Batch Conversion

Convert multiple PKL files at once:

```bash
# Convert all PKL files in current directory
for file in *.pkl; do
    python booster_gym/utils/convert_pkl_for_training.py "$file" "booster_gym/motion_data/t1_4dof/$file"
done
```

## Workflow

1. **Generate Motion Data** (in `general-motion-retargeting` environment):
   ```bash
   conda activate general-motion-retargeting
   python scripts/bvh_to_robot.py --bvh_file dance.bvh --robot booster_t1_4dof --save_path new_dance.pkl
   ```

2. **Convert for Training** (in `general-motion-retargeting` environment):
   ```bash
   python booster_gym/utils/convert_pkl_for_training.py new_dance.pkl booster_gym/motion_data/t1_4dof/new_dance.pkl
   ```

3. **Train the Robot** (in `isaac-gym-training` environment):
   ```bash
   conda activate isaac-gym-training
   cd booster_gym
   python train_imitation.py --task T1Imitation
   ```

## What the Conversion Does

- Converts numpy arrays to Python lists for compatibility
- Uses pickle protocol 2 for cross-version compatibility  
- Handles various data types safely
- Verifies the converted file loads properly
- Creates output directories if needed

## File Structure

```
motion_data/
└── t1_4dof/
    ├── t1_4dof_d1s3.pkl        # Original dancing motion
    ├── new_dance.pkl           # Your new motion
    └── another_motion.pkl      # More motions...
```

## Troubleshooting

If conversion fails:
1. Ensure you're in the `general-motion-retargeting` environment
2. Check that the input PKL file exists and is valid
3. Verify you have write permissions to the output directory
4. Use `--verify` flag to test the converted file

The training system will automatically detect and use all PKL files in the `motion_data/t1_4dof/` directory.