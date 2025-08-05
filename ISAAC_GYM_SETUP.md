# Isaac Gym Installation Guide

This guide helps you install Isaac Gym for the T1 motion imitation training system.

## 🎯 Prerequisites

- **NVIDIA GPU** with CUDA support
- **Python 3.10** (already set up in your conda environment)
- **CUDA Toolkit** (version 11.8 or 12.x recommended)

## 📥 Download Isaac Gym

1. **Register and Download**:
   - Go to: https://developer.nvidia.com/isaac-gym
   - Sign up for NVIDIA Developer account (free)
   - Download **Isaac Gym Preview 4** (latest version)

2. **Extract the Package**:
   ```bash
   # Extract to your preferred location
   tar -xzf IsaacGym_Preview_4_Package.tar.gz
   cd isaacgym
   ```

## 🔧 Installation Steps

### 1. Install Isaac Gym Python Package

```bash
# Activate your conda environment
conda activate general-motion-retargeting

# Install Isaac Gym
cd isaacgym/python
pip install -e .
```

### 2. Configure Environment Variables

Isaac Gym requires specific environment variables for proper GPU access:

```bash
# Add to your conda environment activation script
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
cat > ./etc/conda/activate.d/isaac_gym.sh << 'EOF'
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
EOF

# Add deactivation script
mkdir -p ./etc/conda/deactivate.d
cat > ./etc/conda/deactivate.d/isaac_gym.sh << 'EOF'
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
EOF
```

### 3. Reactivate Environment

```bash
# Deactivate and reactivate to apply changes
conda deactivate
conda activate general-motion-retargeting
```

## ✅ Verify Installation

Test Isaac Gym installation:

```bash
# Test basic import
python -c "import isaacgym; print('✅ Isaac Gym imported successfully')"

# Test GPU access
python -c "
import isaacgym
from isaacgym import gymapi
gym = gymapi.acquire_gym()
print('✅ Isaac Gym GPU access working')
"

# Run Isaac Gym example
cd isaacgym/python/examples
python joint_monkey.py
```

## 🚀 Install Additional Dependencies

Install the remaining dependencies for booster_gym:

```bash
# Install booster_gym requirements
cd /path/to/your/general-motion-retargeting/booster_gym
pip install -r requirements.txt

# Install additional packages for training
pip install tensorboard wandb imageio[ffmpeg]
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Version Mismatch**
```bash
Error: CUDA version mismatch
```
Solution: Ensure your CUDA toolkit version matches PyTorch requirements:
```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvcc --version  # Should match or be compatible
```

**2. Library Loading Issues**
```bash
Error: cannot find shared library libpython3.10.so
```
Solution: Check environment variables are set correctly:
```bash
echo $LD_LIBRARY_PATH
# Should include your conda environment lib path
```

**3. GPU Not Detected**
```bash
Error: No CUDA devices found
```
Solution: Verify NVIDIA drivers and CUDA:
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA version
```

**4. Permission Issues**
```bash
Error: Permission denied accessing GPU
```
Solution: Add user to video group:
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

## 🎯 Test Motion Imitation Training

Once Isaac Gym is installed, test the complete pipeline:

```bash
# Navigate to your project
cd /path/to/general-motion-retargeting

# Test Isaac Gym import with motion imitation
python -c "
import sys
sys.path.append('booster_gym')
import isaacgym
from envs.t1_imitation import T1Imitation
print('✅ T1Imitation environment ready!')
"

# Generate full motion dataset
python scripts/bvh_to_robot_dataset.py \\
    --src_folder lafan1/ \\
    --tgt_folder motion_data/t1_4dof/ \\
    --robot booster_t1_4dof

# Start training (short test)
python booster_gym/train_imitation.py \\
    --task=T1Imitation \\
    --num_envs=512 \\
    --max_iterations=100 \\
    --headless
```

## 📊 Expected Performance

With Isaac Gym properly installed, you should see:

- **Training Speed**: ~1000-2000 FPS with 4096 environments
- **Memory Usage**: ~8-12GB GPU memory for 4096 environments  
- **Convergence**: Visible learning within 1000-2000 iterations

## 🆘 Getting Help

If you encounter issues:

1. **Check Isaac Gym Documentation**: `isaacgym/docs/`
2. **NVIDIA Isaac Gym Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/
3. **Common Solutions**: Check GitHub issues for similar problems

## 🎉 Ready to Train!

Once Isaac Gym is working, you can start training your T1 robot to dance and fight! 🕺🥊

```bash
# Full training command
python booster_gym/train_imitation.py --task=T1Imitation
```

The training will create policies that make your T1 robot imitate human motion from the PKL files we generated! 🚀