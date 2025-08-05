# T1 Motion Imitation System

This document describes the motion imitation system for the Booster T1 humanoid robot, which enables training policies to imitate human motion from mocap data.

## 🎯 Overview

The motion imitation system extends the base Booster Gym locomotion framework to train policies that can replicate human-like movements from motion capture data. The system supports:

- **Full 21-DOF Control**: Arms (8 DOF) + Legs (12 DOF) + Waist (1 DOF)
- **Transfer Learning**: Initialize from pretrained locomotion policies
- **Curriculum Learning**: Progressive training from arms-only to full-body imitation
- **Multiple Motion Formats**: BVH, SMPL-X, and PKL motion files
- **Sim-to-Real Deployment**: Direct deployment to physical T1 robot

## 🏗️ System Architecture

```
BVH/SMPL-X Files → General Motion Retargeting → PKL Files → Motion Imitation Training → Deployed Policy
                   (4-DOF T1 Configuration)              (Isaac Gym + PPO)
```

### Key Components

1. **Motion Loader** (`utils/motion_loader.py`)
   - Loads and preprocesses PKL motion files
   - Provides smooth interpolation and looping
   - Supports motion libraries with multiple sequences

2. **T1Imitation Environment** (`envs/t1_imitation.py`)
   - Extends base T1 environment with motion reference tracking
   - Combines locomotion stability with imitation fidelity
   - Supports 21-DOF control with proper reward balancing

3. **Enhanced Training** (`train_imitation.py`)
   - Transfer learning from pretrained locomotion policies
   - Curriculum learning with configurable stages
   - Advanced reward weighting and parameter freezing

## 🚀 Quick Start

### 1. Generate Motion Data

First, convert your BVH motion files to the T1-compatible PKL format:

```bash
# Generate motion data from LAFAN1 dataset
python scripts/bvh_to_robot_dataset.py --src_folder lafan1/ --tgt_folder motion_data/t1_4dof/ --robot booster_t1_4dof

# Or from custom BVH files
python scripts/bvh_to_robot_dataset.py --src_folder my_bvh_files/ --tgt_folder motion_data/custom/ --robot booster_t1_4dof
```

### 2. Train Imitation Policy

Train a motion imitation policy using the generated data:

```bash
# Basic training (headless by default for performance)
python train_imitation.py --task=T1Imitation

# With custom configuration
python train_imitation.py --task=T1Imitation --num_envs=4096 --headless

# For debugging with visualization
python train_imitation.py --task=T1Imitation --headless=False --num_envs=128
```

**Performance Optimizations (Default Configuration):**
- **Headless Training**: 3x faster performance (~3000+ FPS vs ~1000 FPS)
- **Console Progress**: Real-time training progress with ETA, rewards, and curriculum stages
- **Optimized Horizon**: 64 steps (1.28s planning) for complex motion sequences
- **21→23 DOF Mapping**: PKL data (21 DOF) automatically mapped to full robot (23 DOF)
- **Targeted Learning**: Configurable time segments (e.g., 42s-56s from 2-minute motions)

### 3. Test and Deploy

```bash
# Test in simulation
python play.py --task=T1Imitation --checkpoint=-1

# Export for deployment
python export_model.py --task=T1Imitation --checkpoint=-1

# Deploy to real robot
cd deploy && python deploy.py
```

## ⚙️ Configuration

### Motion Imitation Config (`envs/T1_imitation.yaml`)

Key configuration sections:

```yaml
# Environment setup
env:
  num_actions: 21  # Full 21-DOF control

# Motion data
imitation:
  motion_dir: "motion_data/t1_4dof/"
  loop_motions: true
  imitation_weight: 1.0
  locomotion_weight: 0.1

# Reward scaling
rewards:
  scales:
    # Locomotion rewards (reduced)
    tracking_lin_vel_x: 0.2
    base_height: -5.0
    
    # Imitation rewards (high priority)
    imitation_root_pos: 10.0
    imitation_dof_pos: 15.0
```

### Curriculum Learning

The system supports multi-stage curriculum learning:

```yaml
imitation:
  curriculum_learning: true
  curriculum_stages:
    - name: "arms_only"
      iterations: 2000
      freeze_legs: true
      imitation_weight: 1.0
      locomotion_weight: 0.0
      
    - name: "full_imitation"
      iterations: 13000
      freeze_legs: false
      imitation_weight: 1.0
      locomotion_weight: 0.1
```

## 🎛️ Advanced Usage

### Custom Motion Data

To use your own motion data:

1. **BVH Files**: Place BVH files in a directory and run `scripts/bvh_to_robot_dataset.py`
2. **SMPL-X Files**: Use `scripts/smplx_to_robot_dataset.py` 
3. **Custom PKL**: Create PKL files with the required structure (see Motion Data Format below)

### Transfer Learning

Initialize training from a pretrained locomotion policy:

```yaml
imitation:
  pretrained_policy_path: "deploy/models/T1.pt"
  freeze_legs_initially: true
  unfreeze_after_iterations: 1000
```

### Multi-Motion Training

Train on multiple motion sequences simultaneously:

```yaml
imitation:
  motion_dir: "motion_data/mixed/"  # Directory with multiple PKL files
  resample_motions_on_reset: true   # Sample new motion on episode reset
  time_offset_range: [0.0, 90.0]   # Random start time for 2-minute motions
```

### Targeted Motion Segment Training

Focus training on specific parts of long motion sequences:

```yaml
imitation:
  motion_dir: "motion_data/t1_4dof/"     # PKL files directory
  time_offset_range: [42.0, 42.0]       # Exact segment: 42s-56s
  episode_length_s: 14                   # Match segment length (56-42=14s)
  loop_motions: true                     # Enable looping for consistency
```

**Use Cases:**
- Extract specific choreography from long performances
- Focus on challenging motion segments
- Train on signature moves or key sequences

## 📊 Reward System

The imitation system uses a hybrid reward structure:

### Locomotion Rewards (from base T1)
- **Stability**: Base height, orientation, collision avoidance
- **Energy Efficiency**: Torque minimization, smooth motion
- **Locomotion**: Velocity tracking, foot placement

### Imitation Rewards (new)
- **Position Tracking**: Root position and joint positions
- **Orientation Tracking**: Root orientation matching
- **Velocity Tracking**: Linear and angular velocity matching
- **Smoothness**: Velocity and acceleration consistency

### Reward Weighting Strategy

```python
total_reward = locomotion_weight * locomotion_rewards + imitation_weight * imitation_rewards
```

Typical weights:
- **Early Training**: `locomotion_weight=0.0, imitation_weight=1.0` (arms only)
- **Mixed Training**: `locomotion_weight=0.2, imitation_weight=1.0` 
- **Final Training**: `locomotion_weight=0.1, imitation_weight=1.0`

## 📁 Motion Data Format

PKL files contain motion data with the following structure:

```python
motion_data = {
    "fps": 30,                    # Frame rate
    "root_pos": np.array((N, 3)), # Root position trajectory [x, y, z]
    "root_rot": np.array((N, 4)), # Root rotation [x, y, z, w] quaternion
    "dof_pos": np.array((N, 21)), # Joint positions for all 21 DOF
    "local_body_pos": np.array((N, B, 3)), # Body positions (optional)
    "link_body_list": [...]       # Body names (optional)
}
```

### Joint Ordering (21 DOF)
1. **Arms** (8 DOF): Left/Right Shoulder Pitch/Roll, Elbow Pitch/Yaw
2. **Waist** (1 DOF): Waist Yaw
3. **Legs** (12 DOF): Left/Right Hip Pitch/Roll/Yaw, Knee Pitch, Ankle Pitch/Roll

## 🔧 Troubleshooting

### Common Issues

**1. Motion data not found**
```bash
Error: Motion directory 'motion_data/t1_4dof/' not found
```
Solution: Run `scripts/bvh_to_robot_dataset.py` first to create motion files.

**2. Model dimension mismatch**
```bash
Error: Expected input size 47, got 21
```
Solution: Ensure `num_actions: 21` in your config file for full DOF control.

**3. Training instability**
```bash
Warning: High policy loss, training unstable
```
Solution: Reduce learning rate or increase `locomotion_weight` for stability.

**4. Poor imitation quality**
```bash
Low imitation rewards, robot not following reference
```
Solution: Increase imitation reward scales or reduce reward scaling parameters.

### Performance Optimization

- **GPU Memory**: Reduce `num_envs` if running out of GPU memory (default: 4096)
- **Training Speed**: Headless training enabled by default (~3000+ FPS vs ~1000 FPS)
- **Convergence**: Curriculum learning enabled with cooperative leg-arm training
- **Horizon Length**: Optimized to 64 steps (1.28s) for complex motion planning
- **Monitoring**: Use TensorBoard for local monitoring (wandb disabled by default)

## 📈 Training Tips

### Best Practices

1. **Start Simple**: Begin with single motion sequences before multi-motion training
2. **Use Curriculum**: Enable curriculum learning for complex motions
3. **Monitor Rewards**: Watch both locomotion and imitation reward components
4. **Adjust Scaling**: Tune reward scaling parameters based on motion complexity
5. **Transfer Learning**: Always start from pretrained locomotion policy

### Hyperparameter Tuning

Key parameters to adjust:

- **Learning Rate**: `5e-6` for fine-tuning (default), `1e-5` for scratch training
- **Reward Scales**: Balance between stability (`locomotion_weight`) and fidelity (`imitation_weight`)
- **Episode Length**: Match motion segment length (e.g., `14s` for specific segments)
- **Horizon Length**: `64` steps (1.28s) optimized for complex motion sequences
- **Time Offsets**: Configure `time_offset_range` for targeted segment training

### Training Configuration Examples

**Full Motion Training (2-minute sequences):**
```yaml
episode_length_s: 30
time_offset_range: [0.0, 90.0]  # Random segments
loop_motions: true
```

**Targeted Segment Training:**
```yaml
episode_length_s: 14              # Segment length
time_offset_range: [42.0, 42.0]   # Fixed start time
loop_motions: true
```

**Performance Training:**
```yaml
headless: true                     # 3x faster training
num_envs: 4096                    # Maximum parallelization
horizon_length: 64                 # Optimized planning
use_wandb: false                  # Local monitoring only
console_log_interval: 50          # Console updates every 50 iterations
num_actions: 23                   # Full robot model (21 PKL + 2 head)
record_video: false               # No video overhead during training
```

### Console Progress Monitoring

When running headless, you'll see detailed progress updates like this:

```
================================================================================
[14:23:15] T1 IMITATION TRAINING - Iteration 1,500/20,000 (7.5%)
================================================================================
Progress: [██████░░░░░░░░░░░░░░░░░░░░░░░░] 7.5%
Rewards:  Mean=+2.847 ± 1.234
Curriculum: Stage: arms_focused (75.0%)
Config: Weights: Imitation=1.0, Locomotion=0.8
ETA: 04h 23m remaining
================================================================================
```

**Console Features:**
- **Real-time Progress**: Visual progress bar and percentage
- **Training Metrics**: Mean rewards with standard deviation  
- **Curriculum Status**: Current stage and progress within stage
- **Configuration Info**: Current weight settings
- **Time Estimates**: ETA based on current training speed
- **Automatic Updates**: Every 50 iterations (configurable)

## 🎯 **Optimized Configuration Summary**

The T1Imitation system has been optimized for maximum performance and targeted learning:

### **Key Optimizations:**
- **Headless Training**: 3x performance boost (~3000+ FPS)
- **Cooperative Learning**: Arms and legs train together (no parameter freezing)
- **Targeted Segments**: Train on specific motion parts (e.g., seconds 42-56)
- **Optimized Planning**: 64-step horizon for complex sequences
- **Smart DOF Mapping**: 21-DOF PKL data mapped to 23-DOF robot (head padded with zeros)

### **Curriculum Learning Stages:**
```yaml
Stage 1 (2000 iter): Arms-focused with 80% stability support
Stage 2 (5000 iter): Mixed training with 20% stability  
Stage 3 (13000 iter): Full imitation with 10% stability
```

### **Training Command:**
```bash
# Optimized headless training
python train_imitation.py --task=T1Imitation

# Monitor with TensorBoard
tensorboard --logdir logs
```

## 🚀 Deployment

The trained imitation policies can be deployed directly to the physical T1 robot using the same deployment pipeline as locomotion policies:

```bash
# Export trained model
python export_model.py --task=T1Imitation --checkpoint=-1

# Deploy to robot
cd deploy
python deploy.py --config configs/T1_imitation.yaml
```

The deployment system automatically handles the transition from 21-DOF simulation control to the physical robot's control interface.

## 🔬 Research Extensions

This system provides a foundation for advanced research in:

- **Multi-Modal Imitation**: Combining vision, audio, and motion cues
- **Interactive Imitation**: Real-time motion following and adaptation
- **Style Transfer**: Learning and blending different motion styles
- **Hierarchical Control**: High-level motion planning with low-level imitation
- **Human-Robot Interaction**: Natural motion-based communication

---

For more details, see the main [Booster Gym README](README.md) and the [General Motion Retargeting documentation](../README.md).