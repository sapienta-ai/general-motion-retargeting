import os
import torch
import numpy as np
from typing import Dict, Optional

from .t1 import T1
from utils.motion_loader import MotionLoader, MotionLibrary
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate


class T1Imitation(T1):
    """
    T1 robot environment for motion imitation learning.
    Extends the base T1 locomotion environment with reference motion tracking.
    """
    
    def __init__(self, cfg):
        # Store imitation-specific config
        self.imitation_cfg = cfg.get("imitation", {})
        
        # Call parent constructor first to initialize device
        super().__init__(cfg)
        
        # Initialize motion library after device is available
        self._init_motion_library()
        
        # Initialize imitation-specific buffers
        self._init_imitation_buffers()
        
    def _init_motion_library(self):
        """Initialize the motion library for reference motions."""
        motion_dir = self.imitation_cfg.get("motion_dir", "motion_data/t1_4dof/")
        
        if not os.path.exists(motion_dir):
            raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
            
        self.motion_library = MotionLibrary(
            motion_dir=motion_dir,
            device=self.device,
            motion_files=self.imitation_cfg.get("motion_files", None)
        )
        
        print(f"Loaded {len(self.motion_library.motion_names)} reference motions")
        print(f"🎯 Motion library: {self.motion_library.motion_names}")
        
    def _init_imitation_buffers(self):
        """Initialize buffers for motion imitation tracking."""
        # Reference motion state
        self.ref_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ref_root_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ref_root_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ref_root_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.ref_dof_vel = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        
        # Motion timing and selection
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.motion_lengths = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Assign random motions to each environment
        self.env_motions = []
        for i in range(self.num_envs):
            motion_name = np.random.choice(self.motion_library.motion_names)
            motion = self.motion_library.sample_motion(motion_name)
            self.env_motions.append(motion)
            self.motion_lengths[i] = motion.get_motion_length()
            
        # Motion tracking weights (can be curriculum-based)
        self.imitation_weight = self.imitation_cfg.get("imitation_weight", 1.0)
        self.locomotion_weight = self.imitation_cfg.get("locomotion_weight", 0.1)
        
    def _prepare_reward_function(self):
        """Prepare reward functions including both locomotion and imitation rewards."""
        # Call parent method for locomotion rewards
        super()._prepare_reward_function()
        
        # Add imitation reward functions
        imitation_reward_names = [
            "imitation_root_pos",
            "imitation_root_rot", 
            "imitation_root_vel",
            "imitation_root_ang_vel",
            "imitation_dof_pos",
            "imitation_dof_vel",
            "motion_smoothness",
        ]
        
        # Get imitation reward scales from main rewards config (not imitation config)
        reward_scales = self.cfg.get("rewards", {}).get("scales", {})
        
        for name in imitation_reward_names:
            scale = reward_scales.get(name, 0.0)
            if scale != 0:
                self.reward_scales[name] = scale
                self.reward_functions.append(getattr(self, f"_reward_{name}"))
                self.reward_names.append(name)
                print(f"Added imitation reward: {name} (scale: {scale})")
                
        print(f"Imitation environment prepared with {len(self.reward_functions)} reward functions")
        print(f"📊 Imitation reward scales loaded: {[name for name in imitation_reward_names if name in self.reward_scales]}")
        # Motion library debug will be printed later when it's available
        
    def _update_reference_motion(self):
        """Update reference motion state for all environments."""
        for env_id in range(self.num_envs):
            motion = self.env_motions[env_id]
            motion_time = self.motion_times[env_id].item()
            
            # Get reference state from motion
            ref_state = motion.get_motion_state(motion_time)
            
            # Debug: Print occasionally to verify motion updates
            if hasattr(self, '_motion_debug_counter'):
                self._motion_debug_counter += 1
            else:
                self._motion_debug_counter = 0
                
            if self._motion_debug_counter % 500 == 0 and env_id == 0:  # Debug first env every 500 calls
                print(f"🎬 Motion Update [Env 0]: time={motion_time:.2f}s")
                print(f"   Motion DOF pos (first 8): {ref_state['dof_pos'][:8]}")
                print(f"   Motion DOF vel (first 8): {ref_state['dof_vel'][:8]}")
            
            # Update reference buffers
            self.ref_root_pos[env_id] = ref_state["root_pos"]
            self.ref_root_rot[env_id] = ref_state["root_rot"]  # xyzw
            self.ref_root_vel[env_id] = ref_state["root_vel"]
            self.ref_root_ang_vel[env_id] = ref_state["root_ang_vel"]
            
            # Use full body motion data for dancing (robot has 23 DOFs)
            # Motion data: 21 DOFs (no head), Robot: 23 DOFs (head + arms + waist + legs)
            full_dof_pos = ref_state["dof_pos"]
            full_dof_vel = ref_state["dof_vel"]
            
            if len(full_dof_pos) == 21:  # Motion data has 21 DOFs, robot needs 23
                # CORRECT DOF MAPPING based on T1_serial.urdf:
                # Robot:  [0-1: Head] [2-9: Arms] [10: Waist] [11-22: Legs]
                # Motion: [0-7: Arms] [8: Waist] [9-20: Legs]
                padded_pos = torch.zeros(23, device=self.device)
                padded_vel = torch.zeros(23, device=self.device)
                
                # Map motion DOFs to correct robot DOF indices:
                padded_pos[2:10] = full_dof_pos[0:8]    # Arms: motion[0-7] → robot[2-9]
                padded_vel[2:10] = full_dof_vel[0:8]
                padded_pos[10] = full_dof_pos[8]        # Waist: motion[8] → robot[10]
                padded_vel[10] = full_dof_vel[8]
                
                # 🔧 FIX: Only load PKL leg data if NOT in Stage 1
                if hasattr(self, 'current_stage_name') and self.current_stage_name == "arms_focused":
                    # Stage 1: Keep legs at default positions (don't use PKL leg data)
                    # Legs will be controlled by T1.pt policy, not reference motion
                    if hasattr(self, 'default_dof_pos'):
                        if self.default_dof_pos.dim() == 1:
                            padded_pos[11:23] = self.default_dof_pos[11:23]  # Use default leg positions
                        else:
                            padded_pos[11:23] = self.default_dof_pos.squeeze()[11:23]
                    # Keep leg velocities at zero for Stage 1
                else:
                    # Stage 2+: Use PKL leg data for blending/full imitation
                    padded_pos[11:23] = full_dof_pos[9:21]  # Legs: motion[9-20] → robot[11-22]
                    padded_vel[11:23] = full_dof_vel[9:21]
                    
                # Head joints (robot[0-1]) stay at zero
                self.ref_dof_pos[env_id] = padded_pos
                self.ref_dof_vel[env_id] = padded_vel
                
                # Debug: Print the mapping occasionally
                if self._motion_debug_counter % 500 == 0 and env_id == 0:
                    stage_name = getattr(self, 'current_stage_name', 'unknown')
                    if stage_name == "arms_focused":
                        print(f"   ✅ STAGE 1 - PKL leg data EXCLUDED (using default positions)")
                        print(f"      Arms from PKL: {padded_pos[2:6].cpu().numpy()}")
                        print(f"      Legs DEFAULT:  {padded_pos[11:15].cpu().numpy()}")
                    else:
                        print(f"   ✅ STAGE {stage_name} - PKL leg data INCLUDED")
                        print(f"      Arms from PKL: {padded_pos[2:6].cpu().numpy()}")
                        print(f"      Legs from PKL: {padded_pos[11:15].cpu().numpy()}")
                    print(f"   ✅ FIXED MAPPING:")
                    print(f"   Robot arms[2-9]:  {padded_pos[2:10]}")
                    print(f"   Robot waist[10]:  {padded_pos[10]}")
                    print(f"   Robot head[0-1]:  {padded_pos[0:2]} (zeros)")
            elif len(full_dof_pos) == 23:  # Perfect match
                self.ref_dof_pos[env_id] = full_dof_pos
                self.ref_dof_vel[env_id] = full_dof_vel
            else:
                # Handle other sizes by padding or truncating to 23 DOFs
                padded_pos = torch.zeros(23, device=self.device)
                padded_vel = torch.zeros(23, device=self.device)
                min_dofs = min(len(full_dof_pos), 23)
                padded_pos[:min_dofs] = full_dof_pos[:min_dofs]
                padded_vel[:min_dofs] = full_dof_vel[:min_dofs]
                self.ref_dof_pos[env_id] = padded_pos
                self.ref_dof_vel[env_id] = padded_vel
            
        # Update motion times
        self.motion_times += self.dt
        
        # Reset motions that have finished (if not looping)
        if not self.imitation_cfg.get("loop_motions", True):
            finished_envs = self.motion_times >= self.motion_lengths
            if finished_envs.any():
                self._reset_finished_motions(finished_envs.nonzero(as_tuple=False).flatten())
                
    def _reset_finished_motions(self, env_ids):
        """Reset motions for environments that have finished their sequences."""
        for env_id in env_ids:
            # Sample new motion
            motion_name = np.random.choice(self.motion_library.motion_names)
            motion = self.motion_library.sample_motion(motion_name)
            self.env_motions[env_id.item()] = motion
            
            # Reset timing
            self.motion_times[env_id] = 0.0
            self.motion_lengths[env_id] = motion.get_motion_length()
            
    def _reset_idx(self, env_ids):
        """Reset environments and their motion references."""
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Reset motion timing for reset environments
        for env_id in env_ids:
            # Optionally sample new motion on reset
            if self.imitation_cfg.get("resample_motions_on_reset", True):
                motion_name = np.random.choice(self.motion_library.motion_names)
                motion = self.motion_library.sample_motion(motion_name)
                self.env_motions[env_id.item()] = motion
                self.motion_lengths[env_id] = motion.get_motion_length()
                
            # Reset motion time with optional random offset
            time_offset_range = self.imitation_cfg.get("time_offset_range", [0.0, 0.0])
            time_offset = np.random.uniform(time_offset_range[0], time_offset_range[1])
            self.motion_times[env_id] = time_offset
            
    def _compute_observations(self):
        """Compute observations including motion reference information."""
        # Call parent observation computation
        super()._compute_observations()
        
        # Add reference motion information to observations if configured
        if self.imitation_cfg.get("include_reference_in_obs", False):
            # This would require expanding num_observations in the config
            # For now, we keep the same observation space as the base T1
            pass
            
    # ============ IMITATION REWARD FUNCTIONS ============
    
    def _reward_imitation_root_pos(self):
        """Reward for matching reference root position."""
        pos_error = torch.norm(self.base_pos - self.ref_root_pos, dim=1)
        return torch.exp(-pos_error / self.imitation_cfg.get("pos_reward_scale", 0.5))
        
    def _reward_imitation_root_rot(self):
        """Reward for matching reference root orientation."""
        # Convert root_rot (wxyz) to xyzw for consistency
        current_quat = self.base_quat  # This should be xyzw from IsaacGym
        ref_quat = self.ref_root_rot    # This is xyzw from our motion data
        
        # Compute quaternion difference
        quat_diff = quat_mul(quat_conjugate(current_quat), ref_quat)
        
        # Convert to angle error
        quat_diff_abs = torch.abs(quat_diff)
        angle_error = 2.0 * torch.acos(torch.clamp(quat_diff_abs[:, 3], 0.0, 1.0))
        
        return torch.exp(-angle_error / self.imitation_cfg.get("rot_reward_scale", 0.5))
        
    def _reward_imitation_root_vel(self):
        """Reward for matching reference root velocity."""
        vel_error = torch.norm(self.base_lin_vel - self.ref_root_vel, dim=1)
        return torch.exp(-vel_error / self.imitation_cfg.get("vel_reward_scale", 2.0))
        
    def _reward_imitation_root_ang_vel(self):
        """Reward for matching reference root angular velocity."""
        ang_vel_error = torch.norm(self.base_ang_vel - self.ref_root_ang_vel, dim=1)
        return torch.exp(-ang_vel_error / self.imitation_cfg.get("ang_vel_reward_scale", 1.0))
        
    def _reward_imitation_dof_pos(self):
        """Reward for matching reference joint positions."""
        # Simple direct approach - all DOFs matter equally
        dof_error = torch.norm(self.dof_pos - self.ref_dof_pos, dim=1)
        reward = torch.exp(-dof_error / self.imitation_cfg.get("dof_pos_reward_scale", 1.0))
        
        # Debug print for first environment occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:  # Every 1000 steps
            stage = getattr(self, 'current_stage_name', 'unknown')
            print(f"🔍 DOF Error: {dof_error[0].item():.4f}, Reward: {reward[0].item():.4f} (Stage: {stage})")
            print(f"   Current robot DOFs[0-9]: {self.dof_pos[0, :10]}")
            print(f"   Reference robot DOFs[0-9]: {self.ref_dof_pos[0, :10]}")
            if hasattr(self, 'current_stage_name') and self.current_stage_name == "arms_focused":
                print(f"   ✅ REWARD FOCUS: Only arms[2-9] matter for imitation reward!")
                print(f"   ⚖️  LEGS IGNORED: Legs[11-22] free to stabilize without penalty!")
            print(f"   🎯 TARGET: Robot should move arms[2-9] to match reference!")
            print(f"   🎯 ACTUAL: Arms[2-9] = {self.dof_pos[0, 2:10]}")
            print(f"   🎯 WANTED: Arms[2-9] = {self.ref_dof_pos[0, 2:10]}")
            
        return reward
        
    def _reward_imitation_dof_vel(self):
        """Reward for matching reference joint velocities."""
        # Simple direct approach - all DOFs matter equally
        dof_vel_error = torch.norm(self.dof_vel - self.ref_dof_vel, dim=1)
        return torch.exp(-dof_vel_error / self.imitation_cfg.get("dof_vel_reward_scale", 0.1))
    
    def _reward_motion_smoothness(self):
        """Reward for smooth, natural motion - prevents jerky movements."""
        if not hasattr(self, 'last_dof_vel'):
            return torch.ones(self.num_envs, device=self.device)  # First step
        
        # Calculate joint acceleration (change in velocity)
        joint_accel = torch.norm(self.dof_vel - self.last_dof_vel, dim=1)
        
        # Reward smooth motion, penalize sudden jerky changes
        smoothness_reward = torch.exp(-joint_accel * 0.1)
        return smoothness_reward
        
    def _compute_reward(self):
        """Compute combined locomotion and imitation rewards."""
        # Store original reward buffer
        original_rew = self.rew_buf.clone()
        
        # Compute locomotion rewards (from parent)
        super()._compute_reward()
        locomotion_rew = self.rew_buf.clone()
        
        # Reset reward buffer for imitation rewards
        self.rew_buf[:] = 0.0
        
        # Compute only imitation rewards
        imitation_reward_names = [
            "imitation_root_pos", "imitation_root_rot", "imitation_root_vel",
            "imitation_root_ang_vel", "imitation_dof_pos", "imitation_dof_vel"
        ]
        
        imitation_rew = torch.zeros_like(self.rew_buf)
        for name in imitation_reward_names:
            if name in self.reward_scales:
                rew = getattr(self, f"_reward_{name}")() * self.reward_scales[name]
                imitation_rew += rew
                self.extras["rew_terms"][name] = rew
                
        # Combine rewards with weights
        total_rew = (self.locomotion_weight * locomotion_rew + 
                    self.imitation_weight * imitation_rew)
        
        self.rew_buf[:] = total_rew
        
        # Debug reward components occasionally
        if hasattr(self, '_reward_debug_counter'):
            self._reward_debug_counter += 1
        else:
            self._reward_debug_counter = 0
            
        if self._reward_debug_counter % 2000 == 0:  # Every 2000 steps
            print(f"🎯 Reward Debug [Env 0]:")
            print(f"   Locomotion: {locomotion_rew[0].item():.4f} (weight: {self.locomotion_weight})")
            print(f"   Imitation: {imitation_rew[0].item():.4f} (weight: {self.imitation_weight})")
            print(f"   Total: {total_rew[0].item():.4f}")
            print(f"   Motion time: {self.motion_times[0].item():.2f}s")
        
        # Store reward components for analysis
        self.extras["rew_terms"]["locomotion_total"] = locomotion_rew
        self.extras["rew_terms"]["imitation_total"] = imitation_rew
        
    def step(self, actions):
        """Override step to implement reference motion tracking."""
        # Update reference motion for current timestep FIRST
        self._update_reference_motion()
        
        # Store actions for potential debugging
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        
        # Actions will be used to fine-tune the PKL motion
        
        # No complex staging or T1.pt integration - direct PKL imitation
        
        # 🎯 SIMPLIFIED: Direct PKL imitation with smooth startup transition
        # Let the robot learn its own stabilization through imitation rewards!
        
        # Calculate episode time to handle smooth startup
        episode_time = self.episode_length_buf * self.dt  # Time since episode start
        startup_duration = 2.0  # 2 seconds smooth transition
        
        # Handle different tensor shapes for default_dof_pos
        if self.default_dof_pos.dim() == 1:
            default_pos = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1)
        elif self.default_dof_pos.dim() == 2:
            default_pos = self.default_dof_pos.expand(self.num_envs, -1)
        else:
            default_pos = self.default_dof_pos.squeeze().unsqueeze(0).expand(self.num_envs, -1)
        
        # Smooth startup transition: interpolate from default to PKL over 3 seconds
        in_startup = episode_time < startup_duration
        startup_progress = torch.clamp(episode_time / startup_duration, 0.0, 1.0)  # 0 to 1 over 3 seconds
        
        # Use smooth interpolation (cosine easing for natural movement)
        smooth_factor = 0.5 * (1.0 - torch.cos(startup_progress * 3.14159))  # Smooth S-curve
        
        # Interpolate between default position and PKL reference
        dof_targets = torch.where(
            in_startup.unsqueeze(1),
            default_pos * (1.0 - smooth_factor.unsqueeze(1)) + self.ref_dof_pos * smooth_factor.unsqueeze(1),
            self.ref_dof_pos.clone()  # After 3 seconds, full PKL tracking
        )
        
        # Add small policy contribution for fine-tuning and adaptation
        # Progressive action scaling: optimized for faster learning
        action_scale = torch.where(in_startup, 0.1, 0.2)  # Increased for faster adaptation
        dof_targets += self.cfg["control"]["action_scale"] * self.actions * action_scale.unsqueeze(1)
        
        # Debug: Show what we're doing occasionally
        if hasattr(self, '_simple_debug_counter'):
            self._simple_debug_counter += 1
        else:
            self._simple_debug_counter = 0
            
        if self._simple_debug_counter % 2000 == 0:  # Every 2000 steps
            env_id = 0  # First environment
            is_startup = in_startup[env_id].item()
            current_progress = startup_progress[env_id].item()
            current_smooth = smooth_factor[env_id].item()
            
            if is_startup:
                print(f"🚀 SMOOTH STARTUP [Env {env_id}] - {episode_time[env_id]:.2f}s / {startup_duration}s:")
                print(f"   Progress:       {current_progress:.2%} (smooth factor: {current_smooth:.3f})")
                print(f"   Action scale:   {action_scale[env_id]:.3f} (conservative during startup)")
                print(f"   Default arms:   {default_pos[env_id, 2:6].cpu().numpy()}")
                print(f"   PKL arms:       {self.ref_dof_pos[env_id, 2:6].cpu().numpy()}")
                print(f"   Target arms:    {dof_targets[env_id, 2:6].cpu().numpy()}")
                print(f"   Current arms:   {self.dof_pos[env_id, 2:6].cpu().numpy()}")
            else:
                print(f"🎯 FULL PKL IMITATION [Env {env_id}] - {episode_time[env_id]:.2f}s:")
                print(f"   Action scale:   {action_scale[env_id]:.3f} (normal)")
                print(f"   PKL arms:       {self.ref_dof_pos[env_id, 2:6].cpu().numpy()}")
                print(f"   Target arms:    {dof_targets[env_id, 2:6].cpu().numpy()}")
                print(f"   Current arms:   {self.dof_pos[env_id, 2:6].cpu().numpy()}")
                print(f"   PKL legs:       {self.ref_dof_pos[env_id, 11:15].cpu().numpy()}")
                print(f"   Target legs:    {dof_targets[env_id, 11:15].cpu().numpy()}")
                print(f"   Current legs:   {self.dof_pos[env_id, 11:15].cpu().numpy()}")
            print(f"   Action contrib: {(self.actions[env_id, 11:15] * action_scale[env_id]).cpu().numpy()}")
        
        # Perform physics simulation (copied from parent)
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # Post physics step (copied from parent)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Continue with parent logic (copied from T1.step)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras