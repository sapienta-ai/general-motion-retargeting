#!/usr/bin/env python3

"""
Training script for T1 motion imitation with transfer learning support.
Supports loading pretrained locomotion policies and curriculum learning.
"""

import os
import numpy as np
import isaacgym
import torch
from utils.runner import Runner
from utils.model import ActorCritic
from utils.recorder import Recorder


class ImitationRunner(Runner):
    """
    Enhanced runner for motion imitation training with transfer learning.
    """
    
    def __init__(self, test=False):
        super().__init__(test)
        
        # Set recorder class for training loop  
        self.recorder_class = Recorder
        
        # Initialize training parameters missing from parent class
        self.batch_size = min(self.cfg["runner"]["horizon_length"] * self.env.num_envs // 4, 4096)
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = self.cfg["algorithm"]["entropy_coef"] 
        self.bound_loss_coef = self.cfg["algorithm"]["bound_coef"]
        self.max_grad_norm = 1.0
        
        # Add missing buffers for imitation training
        self.buffer.add_buffer("values", ())
        self.buffer.add_buffer("advantages", ())
        self.buffer.add_buffer("returns", ())
        
        # Initialize curriculum learning (always needed)
        self._setup_curriculum_learning()
        
        # Apply first curriculum stage at start
        if self.curriculum_stages and len(self.curriculum_stages) > 0:
            first_stage = self.curriculum_stages[0]
            self._update_stage_config(first_stage)
            print(f"🎯 Starting with curriculum stage 0: {first_stage['name']}")
        
        # Initialize transfer learning if configured
        if 'imitation' in self.cfg and not test:
            self._setup_transfer_learning()
            
        # Pass runner reference to environment for locomotion policy access
        self._link_runner_to_env()
        
        # Store current iteration for locomotion policy
        self.current_iteration = 0
        
    def _link_runner_to_env(self):
        """Link the runner instance to environments for locomotion policy access."""
        try:
            # Direct environment access - in Isaac Gym, self.env is the T1Imitation instance
            self.env._runner = self
            print("✅ Linked runner to environment for locomotion policy access")
        except Exception as e:
            print(f"⚠️  Warning: Could not link runner to environment: {e}")
            print("   Locomotion policy integration may not work properly")
    
    def _setup_transfer_learning(self):
        """Setup transfer learning from pretrained locomotion policy."""
        imitation_cfg = self.cfg.get('imitation', {})
        pretrained_path = imitation_cfg.get('pretrained_policy_path', None)
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained policy from: {pretrained_path}")
            
            try:
                # Load the pretrained JIT model and move to correct device
                pretrained_model = torch.jit.load(pretrained_path)
                pretrained_model.eval()
                pretrained_model = pretrained_model.to(self.device)
                
                # Extract state dict (this is tricky with JIT models)
                # We'll need to manually initialize our model with similar weights
                self._initialize_from_pretrained(pretrained_model)
                
                print("Successfully initialized from pretrained locomotion policy")
                
            except Exception as e:
                print(f"Warning: Could not load pretrained policy: {e}")
                print("Starting training from scratch")
        else:
            print("No pretrained policy specified or file not found")
            
    def _initialize_from_pretrained(self, pretrained_model):
        """
        Initialize current model weights from pretrained model.
        This loads the pretrained T1.pt locomotion policy for leg control.
        """
        try:
            # Store the pretrained model for leg action generation
            self.pretrained_policy = pretrained_model
            self.pretrained_policy.eval()
            
            # Initialize locomotion policy variables (from policy.py)
            self.locomotion_obs = torch.zeros(47, dtype=torch.float32, device=self.device)
            self.locomotion_actions = torch.zeros(12, dtype=torch.float32, device=self.device)
            self.gait_frequency = 1.0
            self.smoothed_commands = torch.zeros(3, dtype=torch.float32, device=self.device)
            self.last_locomotion_actions = torch.zeros(12, dtype=torch.float32, device=self.device)
            
            # Policy configuration from deploy config
            self.policy_normalization = {
                'gravity': 1.0,
                'ang_vel': 1.0, 
                'lin_vel': 1.0,
                'dof_pos': 1.0,
                'dof_vel': 0.1,
                'clip_actions': 1.0
            }
            
            print("✅ Successfully loaded pretrained T1.pt locomotion policy for leg stabilization")
            
        except Exception as e:
            print(f"❌ Error loading pretrained policy: {e}")
            self.pretrained_policy = None
        
    def _print_training_progress(self, iteration, rewards):
        """Print detailed training progress to console for headless monitoring."""
        import time
        
        # Get reward statistics
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        
        # Get current curriculum stage info
        stage_info = ""
        if hasattr(self, 'current_stage') and hasattr(self, 'curriculum_stages'):
            if self.current_stage < len(self.curriculum_stages):
                stage = self.curriculum_stages[self.current_stage]
                stage_progress = (iteration - self.stage_start_iteration) / stage['iterations'] * 100
                stage_info = f"Stage: {stage['name']} ({stage_progress:.1f}%)"
            else:
                stage_info = "Stage: Curriculum Complete"
        
        # Get environment-specific info
        env_info = ""
        if hasattr(self.env, 'imitation_weight') and hasattr(self.env, 'locomotion_weight'):
            env_info = f"Weights: Imitation={self.env.imitation_weight:.1f}, Locomotion={self.env.locomotion_weight:.1f}"
        
        # Progress bar
        max_iter = self.cfg["basic"]["max_iterations"]
        progress = iteration / max_iter * 100
        bar_length = 30
        filled_length = int(bar_length * iteration // max_iter)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Get timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # Print comprehensive progress
        print(f"\n{'='*80}")
        print(f"[{timestamp}] T1 IMITATION TRAINING - Iteration {iteration:,}/{max_iter:,} ({progress:.1f}%)")
        print(f"{'='*80}")
        print(f"Progress: [{bar}] {progress:.1f}%")
        print(f"Rewards:  Mean={reward_mean:+.3f} ± {reward_std:.3f}")
        if stage_info:
            print(f"Curriculum: {stage_info}")
        if env_info:
            print(f"Config: {env_info}")
        
        # Estimate remaining time
        if iteration > 0:
            if not hasattr(self, '_start_time'):
                self._start_time = time.time()
            elapsed = time.time() - self._start_time
            time_per_iter = elapsed / iteration
            remaining_iters = max_iter - iteration
            eta_seconds = remaining_iters * time_per_iter
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            print(f"ETA: {eta_hours:02d}h {eta_mins:02d}m remaining")
        
        print(f"{'='*80}\n")
        
        # Force flush output for immediate console display
        import sys
        sys.stdout.flush()
        
    def _setup_curriculum_learning(self):
        """Setup curriculum learning stages."""
        imitation_cfg = self.cfg.get('imitation', {})
        
        # Always initialize curriculum attributes
        self.curriculum_stages = []
        self.current_stage = 0
        self.stage_start_iteration = 0
        
        if imitation_cfg.get('curriculum_learning', False):
            self.curriculum_stages = imitation_cfg.get('curriculum_stages', [])
            
            print(f"Curriculum learning enabled with {len(self.curriculum_stages)} stages")
            for i, stage in enumerate(self.curriculum_stages):
                print(f"  Stage {i}: {stage['name']} ({stage['iterations']} iterations)")
        else:
            print("Curriculum learning disabled")
            
    def _update_curriculum(self, iteration):
        """Update curriculum learning stage based on current iteration."""
        if not self.curriculum_stages:
            return
            
        # Check if we should advance to next stage
        if self.current_stage < len(self.curriculum_stages):
            current_stage_config = self.curriculum_stages[self.current_stage]
            iterations_in_stage = iteration - self.stage_start_iteration
            
            if iterations_in_stage >= current_stage_config['iterations']:
                # Advance to next stage
                self.current_stage += 1
                self.stage_start_iteration = iteration
                
                if self.current_stage < len(self.curriculum_stages):
                    next_stage = self.curriculum_stages[self.current_stage]
                    print(f"\\n=== Advancing to curriculum stage {self.current_stage}: {next_stage['name']} ===")
                    
                    # Update environment reward weights
                    self._update_stage_config(next_stage)
                else:
                    print("\\n=== Curriculum learning completed ===")
                    
    def _update_stage_config(self, stage_config):
        """Update environment configuration for current curriculum stage."""
        # Update reward weights in the environment
        if hasattr(self.env, 'imitation_weight'):
            self.env.imitation_weight = stage_config.get('imitation_weight', 1.0)
            self.env.locomotion_weight = stage_config.get('locomotion_weight', 0.1)
            # Set current stage name for progressive reference tracking
            self.env.current_stage_name = stage_config["name"]
            print(f"📊 Updated weights: Imitation={self.env.imitation_weight}, Locomotion={self.env.locomotion_weight}")
            print(f"🎯 Stage focus: {stage_config['name']} - Progressive DOF tracking enabled")
            
        # Handle parameter freezing (simplified)
        freeze_legs = stage_config.get('freeze_legs', False)
        if freeze_legs != getattr(self, '_legs_frozen', False):
            self._toggle_leg_freezing(freeze_legs)
            
    def _toggle_leg_freezing(self, freeze: bool):
        """Toggle freezing of leg-related parameters."""
        # This is a simplified implementation
        # In practice, you'd identify which parameters correspond to legs
        # and set their requires_grad accordingly
        
        self._legs_frozen = freeze
        
        if freeze:
            print("Freezing leg parameters for curriculum learning")
            # Identify and freeze leg parameters
            # for param_name, param in self.model.named_parameters():
            #     if 'leg' in param_name.lower():  # Simple heuristic
            #         param.requires_grad = False
        else:
            print("Unfreezing all parameters")
            # Unfreeze all parameters
            for param in self.model.parameters():
                param.requires_grad = True
                
    def generate_locomotion_targets(self, env_batch_data, iteration):
        """
        Generate DOF targets using T1.pt policy EXACTLY like in deployment.
        Returns full 23-DOF targets (not just leg actions).
        """
        if not hasattr(self, 'pretrained_policy') or self.pretrained_policy is None:
            return None
        
        try:
            batch_size = env_batch_data['dof_pos'].shape[0]
            dof_targets_batch = torch.zeros(batch_size, 23, dtype=torch.float32, device=self.device)
            
            # Initialize state variables if not exists
            if not hasattr(self, 'policy_state'):
                self.policy_state = {
                    'gait_process': 0.0,
                    'gait_frequency': 1.0,
                    'commands': torch.zeros(3, dtype=torch.float32, device=self.device),
                    'smoothed_commands': torch.zeros(3, dtype=torch.float32, device=self.device),
                    'last_actions': torch.zeros(12, dtype=torch.float32, device=self.device),
                    'policy_interval': 0.02  # 50Hz
                }
            
            # Use first environment for policy (broadcast to all)
            env_id = 0
            dof_pos = env_batch_data['dof_pos'][env_id]  # [23]
            dof_vel = env_batch_data['dof_vel'][env_id]  # [23] 
            base_ang_vel = env_batch_data['base_ang_vel'][env_id]  # [3]
            projected_gravity = env_batch_data['projected_gravity'][env_id]  # [3]
            
            # Default DOF positions (EXACT deployment config)
            default_dof_pos = torch.tensor([
                0.0, 0.0,                                    # head
                0.2, -1.35, 0.0, -0.5, 0.2, 1.35, 0.0, 0.5, # arms (deployment pose)
                0.0,                                         # waist
                -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,           # left leg (deployment pose)
                -0.2, 0.0, 0.0, 0.4, -0.25, 0.0            # right leg (deployment pose)
            ], dtype=torch.float32, device=self.device)
            
            # Update gait process
            time_now = iteration * 0.02  # Assume 50Hz
            self.policy_state['gait_process'] = (time_now * self.policy_state['gait_frequency']) % 1.0
            
            # Commands (deployment-style: mostly zero with tiny variation)
            # Very small commands to mimic deployment standing behavior
            time_phase = (iteration * 0.02) % 30.0  # Slow 30 second cycle
            self.policy_state['commands'][0] = 0.005 * torch.sin(torch.tensor(time_phase * 0.05, device=self.device))  # Minimal variation
            self.policy_state['commands'][1] = 0.0  # vy (always zero like deployment)
            self.policy_state['commands'][2] = 0.003 * torch.cos(torch.tensor(time_phase * 0.03, device=self.device))  # Minimal rotation
            
            # Smooth commands (like in deploy)
            clip_range = self.policy_state['policy_interval']
            command_diff = self.policy_state['commands'] - self.policy_state['smoothed_commands']
            self.policy_state['smoothed_commands'] += torch.clamp(command_diff, -clip_range, clip_range)
            
            # Update gait frequency based on commands
            if torch.norm(self.policy_state['smoothed_commands']) < 1e-5:
                self.policy_state['gait_frequency'] = 0.0
            else:
                self.policy_state['gait_frequency'] = 1.0
            
            # Build observation vector EXACTLY like deploy
            obs = torch.zeros(47, dtype=torch.float32, device=self.device)
            
            # Normalization constants (from deploy config)
            gravity_norm = 1.0
            ang_vel_norm = 1.0
            lin_vel_norm = 1.0
            dof_pos_norm = 1.0
            dof_vel_norm = 0.1
            
            obs[0:3] = projected_gravity * gravity_norm
            obs[3:6] = base_ang_vel * ang_vel_norm
            
            gait_active = self.policy_state['gait_frequency'] > 1e-8
            obs[6] = self.policy_state['smoothed_commands'][0] * lin_vel_norm * gait_active
            obs[7] = self.policy_state['smoothed_commands'][1] * lin_vel_norm * gait_active  
            obs[8] = self.policy_state['smoothed_commands'][2] * ang_vel_norm * gait_active
            obs[9] = torch.cos(2 * torch.pi * torch.tensor(self.policy_state['gait_process'], device=self.device)) * gait_active
            obs[10] = torch.sin(2 * torch.pi * torch.tensor(self.policy_state['gait_process'], device=self.device)) * gait_active
            
            # DOF observations (legs only, indices 11:23)
            obs[11:23] = (dof_pos[11:23] - default_dof_pos[11:23]) * dof_pos_norm
            obs[23:35] = dof_vel[11:23] * dof_vel_norm
            
            # Previous actions
            obs[35:47] = self.policy_state['last_actions']
            
            # Run policy
            with torch.no_grad():
                actions = self.pretrained_policy(obs.unsqueeze(0)).squeeze(0)
                actions = torch.clamp(actions, -1.0, 1.0)  # Clip actions
                
                # Store for next iteration
                self.policy_state['last_actions'] = actions
                
                # Generate DOF targets EXACTLY like deploy
                # Start with default positions for ALL DOFs
                dof_targets = default_dof_pos.clone()
                
                # Only modify legs (indices 11:23) with policy actions (DEPLOYMENT EXACT)
                dof_targets[11:23] += 1.0 * actions  # EXACT deployment action scale
                
                # Broadcast to all environments
                for i in range(batch_size):
                    dof_targets_batch[i] = dof_targets
                
                return dof_targets_batch
            
        except Exception as e:
            print(f"Error in T1.pt policy integration: {e}")
            return None
                
    def train(self):
        """Enhanced training loop with curriculum learning."""
        self.recorder = self.recorder_class(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        
        # Initialize timing for console progress
        import time
        self._start_time = time.time()
        print(f"\n🚀 Starting T1 Imitation Training ({self.cfg['basic']['max_iterations']:,} iterations)")
        print(f"📍 Target motion segment: 0s-28s (28 seconds) with 2s smooth startup - SPEED OPTIMIZED")
        print(f"⚙️  Configuration: {self.cfg['env']['num_envs']:,} envs, headless={self.cfg['basic']['headless']}")
        print(f"🎯 Curriculum stages: {len(getattr(self, 'curriculum_stages', []))} stages configured")
        print("="*80 + "\n")
        
        for it in range(self.cfg["basic"]["max_iterations"]):
            # Update current iteration for locomotion policy
            self.current_iteration = it
            
            # Update curriculum learning
            self._update_curriculum(it)
            
            # Regular training loop (same as parent)
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                    
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(
                    done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1)
                )

            # Policy update - compute values and advantages first
            with torch.no_grad():
                # Compute values for all states
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                last_values = self.model.est_value(obs, privileged_obs)
                
                # Compute advantages using GAE
                from utils.utils import discount_values
                self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    values,
                    last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Store in buffers - these are computed for the entire rollout
                for t in range(self.cfg["runner"]["horizon_length"]):
                    self.buffer.update_data("values", t, values[t])
                    self.buffer.update_data("advantages", t, advantages[t])
                    self.buffer.update_data("returns", t, returns[t])
            
            # Console Progress Reporting
            log_interval = self.cfg.get("runner", {}).get("console_log_interval", 50)
            if it % log_interval == 0 or it < 100:  # More frequent early, then every N iterations
                self._print_training_progress(it, rew)
                
            # Compute old action log probabilities (detached from computational graph)
            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            
            for n in range(self.cfg["runner"]["mini_epochs"]):
                batch_indices = torch.randperm(
                    self.cfg["runner"]["horizon_length"] * self.env.num_envs,
                    dtype=torch.long,
                    device=self.device
                )
                
                for start in range(0, len(batch_indices), self.batch_size):
                    end = start + self.batch_size
                    batch_idx = batch_indices[start:end]
                    
                    obs_batch = self.buffer["obses"].view(-1, self.env.num_obs)[batch_idx]
                    privileged_obs_batch = self.buffer["privileged_obses"].view(-1, self.env.num_privileged_obs)[batch_idx]
                    actions_batch = self.buffer["actions"].view(-1, self.env.num_actions)[batch_idx]
                    target_values_batch = self.buffer["values"].view(-1)[batch_idx]
                    advantages_batch = self.buffer["advantages"].view(-1)[batch_idx]
                    returns_batch = self.buffer["returns"].view(-1)[batch_idx]
                    old_actions_log_prob_batch = old_actions_log_prob.view(-1)[batch_idx]

                    dist_batch = self.model.act(obs_batch)
                    actions_log_prob_batch = dist_batch.log_prob(actions_batch).sum(dim=-1)
                    value_batch = self.model.est_value(obs_batch, privileged_obs_batch)
                    entropy_batch = dist_batch.entropy().sum(dim=-1)

                    # KL divergence
                    approx_kl = (old_actions_log_prob_batch - actions_log_prob_batch).mean()

                    # Policy loss
                    ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
                    surr1 = advantages_batch * ratio
                    surr2 = advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                    # Entropy loss
                    entropy_loss = entropy_batch.mean()

                    # Bound loss
                    bound_loss = self.bound_loss_coef * (dist_batch.loc.abs().mean() + dist_batch.scale.mean())

                    loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss + bound_loss

                    # Gradient step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    mean_value_loss += value_loss.item()
                    mean_actor_loss += actor_loss.item()
                    mean_bound_loss += bound_loss.item()
                    mean_entropy += entropy_loss.item()

            # Logging and saving (same as parent)
            
            num_updates = self.cfg["runner"]["mini_epochs"] * len(batch_indices) // self.batch_size
            mean_value_loss /= num_updates
            mean_actor_loss /= num_updates
            mean_bound_loss /= num_updates
            mean_entropy /= num_updates

            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl": approx_kl.item(),
                    "curriculum_stage": self.current_stage if hasattr(self, 'current_stage') else 0,
                }, it
            )

            if it % self.cfg["runner"]["save_interval"] == 0:
                model_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "curriculum_stage": self.current_stage if hasattr(self, 'current_stage') else 0,
                }
                self.recorder.save(model_dict, it)

            self.recorder.step()

        # Training completion message
        import time
        elapsed_time = time.time() - self._start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        print(f"\n🎉 T1 Imitation Training Complete!")
        print(f"⏱️  Total time: {hours:02d}h {minutes:02d}m")
        print(f"📊 Final reward: {rew.mean().item():+.3f}")
        print(f"💾 Models saved to: {self.recorder.model_dir}")
        print("🚀 Ready for deployment!")

        self.recorder.close()


if __name__ == "__main__":
    # Use the enhanced runner for imitation training
    runner = ImitationRunner(test=False)
    runner.train()