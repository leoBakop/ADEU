import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from network import PPONetwork, PPO_RNDModel
import torch.nn.functional as F



class PPO_Agent(nn.Module):
    def __init__(self, 
                 envs,
                 num_steps,
                 gamma,
                 batch_size,
                 minibatch_size,
                 gae_lambda,
                 update_epochs,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 learning_rate,
                 num_iterations,
                 anneal_lr,
                 device,
                 head_start,
                 **kwargs):
        
        super().__init__()

        self.network = PPONetwork(envs)


        self.device = device
        self.num_steps = num_steps
        self.envs = envs
        self.gamma = gamma
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.clip_coef = 0.1
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.head_start = head_start
        self.num_actions = envs.single_action_space.n
        self.training_steps = 0
        self.update_epochs = update_epochs
        self.norm_adv = True
        self.clip_vloss = True

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def get_variance(self, ):
        return 0, 0

    def get_value(self, x):
        return self.network(x, return_critic = True)

    def get_action_and_value(self, x, action=None):
        logits = self.network(x, return_actor = True)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.network(x, return_critic = True)
    

    def get_clear_action(self, x:np.array, evaluate = True):

        if self.training_steps < self.head_start:
            if evaluate: return np.random.choice(self.num_actions)
            return np.random.choice(self.num_actions, size=x.shape[0])

        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self.network(x, return_actor = True)

        if evaluate:  
            return np.argmax(logits.cpu().detach().numpy()) 
        return np.argmax(logits.cpu().detach().numpy(), axis=1) 

        
        
    

    def reduce_lr(self, iteration, ):
        if self.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.num_iterations
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

    def train(self, obss, actions, rewards, logprobs, dones, values, next_obs, next_done):
        self.training_steps+=1

        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obss.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return loss


class RewardForwardFilter:
    """EMA filter used in CleanRL RND to produce more stable intrinsic returns."""
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems



class PPO_RND_Agent:
    """
    Composed agent that:
      - owns a PPO_Agent (policy + ext/int critics)
      - owns an RNDModel (predictor + target)
      - computes intrinsic rewards, normalizes them, computes separate GAEs
      - runs PPO updates using combined advantage = int_coef * int_adv + ext_coef * ext_adv
      - trains RND predictor at the same frequency (masked proportion per minibatch)
    """

    def __init__(self,
                 ppo_agent: PPO_Agent,
                 rnd_model: PPO_RNDModel,
                 device,
                 update_proportion=0.25,
                 int_coef=1.0,
                 ext_coef=2.0,
                 int_gamma=0.99):
        self.ppo_agent = ppo_agent
        self.rnd = rnd_model.to(device)
        self.device = device
        self.update_proportion = update_proportion
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.int_gamma = int_gamma
        self.anneal_lr = ppo_agent.anneal_lr
        self.num_iterations = ppo_agent.num_iterations
        self.learning_rate = ppo_agent.learning_rate
        # combined optimizer: PPO parameters + predictor parameters (target is frozen)
        combined_parameters = list(self.ppo_agent.parameters()) + list(self.rnd.predictor.parameters())
        self.optimizer = optim.Adam(combined_parameters, lr=self.ppo_agent.learning_rate, eps=1e-5)

        # running stats for intrinsic rewards (target normalization)
        from gym.wrappers.normalize import RunningMeanStd
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.discounted_reward = RewardForwardFilter(self.int_gamma)
        self.variance = 0
        self.episodes_without_evaluation = 0


    def reduce_lr(self, iteration, ):
        if self.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.num_iterations
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow


    def get_clear_action(self, x):
        return self.ppo_agent.get_clear_action(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.ppo_agent.network(x, return_actor = True)
        probs = Categorical(logits=logits)
        if action is None:
            #action = probs.sample()
            action = torch.argmax(logits, axis = 1)
        return action, probs.log_prob(action), probs.entropy(), self.ppo_agent.network(x, return_critic = True)


    def get_variance(self, normilized = True):
        if normilized:
            v = self.variance/(self.episodes_without_evaluation+1)
            self.episodes_without_evaluation = 0
            return v, 0
        return self.variance, 0

    def compute_intrinsic(self, next_obs):
        """
        next_obs: torch tensor (num_envs, C, H, W)
        we extract the last channel and normalize using obs_rms then forward through rnd networks.
        returns: intrinsic rewards tensor shape (num_envs,)
        """
        # extract same as original (channel index 3)
        rnd_input = (next_obs[:, 3, :, :].reshape(next_obs.shape[0], 1, 84, 84) - torch.from_numpy(self.obs_rms.mean).to(self.device)) / torch.sqrt(torch.from_numpy(self.obs_rms.var).to(self.device))
        rnd_input = rnd_input.clamp(-5, 5).float()
        with torch.no_grad():
            target_feat = self.rnd.target(rnd_input)
            pred_feat = self.rnd.predictor(rnd_input)
            intrinsic = ((target_feat - pred_feat).pow(2).sum(1) / 2.0).cpu().numpy()
        return intrinsic, rnd_input

    def update_obs_rms(self, obs_array):
        # obs_array: numpy array of shape (N, 1, 84, 84)
        self.obs_rms.update(obs_array)

    def train(self,
              obs,             # tensor (num_steps, num_envs, C, H, W)
              actions,         # tensor (num_steps, num_envs) or with action dim
              ext_rewards,     # tensor (num_steps, num_envs)
              logprobs,        # tensor (num_steps, num_envs)
              dones,           # tensor (num_steps, num_envs)
              ext_values,      # tensor (num_steps, num_envs)
              next_obs,        # tensor (num_envs, C, H, W)
              next_done,       # tensor (num_envs,)
              ):
        """
        Performs:
          - compute intrinsic rewards for the rollout (and normalize)
          - compute ext and int GAEs and returns
          - combine advantages for policy updates
          - run minibatch PPO updates and RND predictor updates (same frequency)
        Returns:
          (ppo_loss_item, rnd_forward_loss_item)
        """

        self.episodes_without_evaluation+=1

        num_steps, num_envs = obs.shape[0], obs.shape[1]
        device = self.device

        # 1) compute intrinsic rewards per step for all next states in rollout
        # We compute intrinsic for the sequence of "next" observations corresponding to each step.
        # For step t, next_obs_for_step is obs[t+1] except last step uses provided next_obs
        intrinsic_rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        rnd_inputs_all = []

        # Build "next states" for curiosity consistent with transitions
        for t in range(num_steps):
            if t < num_steps - 1:
                next_obs_t = obs[t + 1].to(device)
            else:
                next_obs_t = next_obs.to(device)
            intrinsic_np, rnd_input = self.compute_intrinsic(next_obs_t)
            intrinsic_rewards[t] = torch.tensor(intrinsic_np, device=device)
            rnd_inputs_all.append(rnd_input.detach().cpu().numpy())

        # process intrinsic normalization similar to original script
        # discounted intrinsic per env
        intrinsic_per_env = np.array([self.discounted_reward.update(rw) for rw in intrinsic_rewards.cpu().data.numpy().T])
        mean, std, count = np.mean(intrinsic_per_env), np.std(intrinsic_per_env), len(intrinsic_per_env)
        self.reward_rms.update_from_moments(mean, std**2, count)
        intrinsic_rewards = intrinsic_rewards / np.sqrt(self.reward_rms.var)

        # Update observation RMS using all rnd_inputs collected (stack)
        rnd_inputs_stack = np.concatenate(rnd_inputs_all, axis=0)  # shape (num_steps*num_envs, 1, 84,84)
        self.obs_rms.update(rnd_inputs_stack)

        # 2) compute separate GAEs for extrinsic and intrinsic returns
        # ext_values provided correspond to external critic; assume internal critic is also produced by PPO network in get_action_and_value.
        # for internal values we need to query the actor/critic - we will reuse ppo_agent.get_value applied to obs to get int_values.
        with torch.no_grad():
            # compute int_values from agent's critic (we assume network has a second critic when requested,
            # but user's PPO_Agent.get_value may return appropriate tensor - if not, adapt to your PPONetwork)
            # Here, we attempt to get internal value via ppo_agent.get_value (which in your previous form returned ext and int).
            # If get_value returns a single value, we will reuse ext_values for int_values (best-effort).
            try:
                # Try to get both critics from the network if available via agent.get_value
                val = self.ppo_agent.get_value(next_obs.to(device))
                # If val is tuple-like (ext, int), handle accordingly
                if isinstance(val, tuple) or (hasattr(val, 'shape') and val.ndim > 1 and val.shape[0] > 1):
                    next_ext_value = val[0].reshape(1, -1)
                    next_int_value = val[1].reshape(1, -1)
                else:
                    next_ext_value = val.reshape(1, -1)
                    next_int_value = val.reshape(1, -1)
            except Exception:
                # fallback: use ext_values last step for both
                next_ext_value = ext_values[-1].reshape(1, -1)
                next_int_value = ext_values[-1].reshape(1, -1)

            # Prepare advantage holders
            ext_advantages = torch.zeros_like(ext_rewards).to(device)
            int_advantages = torch.zeros_like(intrinsic_rewards).to(device)
            ext_lastgaelam = 0
            int_lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_ext_value
                    int_nextvalues = next_int_value
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    # for int_nextvalues use int_values buffer if available; attempt to get via network otherwise reuse ext
                    try:
                        # compute int_values from network on obs[t+1]
                        int_nextvalues = self.ppo_agent.get_value(obs[t + 1].to(device))[1]
                    except Exception:
                        int_nextvalues = ext_values[t + 1]

                ext_delta = ext_rewards[t] + self.ppo_agent.gamma * ext_nextvalues.cpu() * ext_nextnonterminal - ext_values[t]
                int_delta = intrinsic_rewards[t] + self.int_gamma * int_nextvalues * int_nextnonterminal - (self.ppo_agent.get_value(obs[t].to(device))[1] if True else 0)
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + self.ppo_agent.gamma * self.ppo_agent.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + self.int_gamma * self.ppo_agent.gae_lambda * int_nextnonterminal * int_lastgaelam
                )

            ext_returns = ext_advantages.cpu() + ext_values
            int_returns = int_advantages.cpu() + (ext_values)  # if separate int-values exist, replace ext_values accordingly

        # 3) flatten batch for training
        b_obs = obs.reshape((-1,) + tuple(obs.shape[2:])).to(device)
        b_actions = actions.reshape((-1,) + tuple(actions.shape[2:])).to(device) if actions.ndim > 2 else actions.reshape(-1).to(device)
        b_logprobs = logprobs.reshape(-1).to(device)
        b_ext_adv = ext_advantages.reshape(-1).to(device)
        b_int_adv = int_advantages.reshape(-1).to(device)
        b_ext_returns = ext_returns.reshape(-1).to(device)
        b_int_returns = int_returns.reshape(-1).to(device)
        b_ext_values = ext_values.reshape(-1).to(device)

        # combined advantage for policy
        b_advantages = b_int_adv * self.int_coef + b_ext_adv * self.ext_coef

        # 4) PPO + RND optimization loop (minibatches + epochs)
        batch_size = self.ppo_agent.batch_size
        minibatch_size = self.ppo_agent.minibatch_size
        clip_coef = self.ppo_agent.clip_coef

        b_inds = np.arange(batch_size)
        ppo_loss_val = None
        rnd_forward_loss = None

        for epoch in range(self.ppo_agent.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # get minibatch tensors
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_ext_returns = b_ext_returns[mb_inds]
                mb_ext_values = b_ext_values[mb_inds]

                # evaluate current policy on minibatch
                _, newlogprob, entropy, newvalue = self.ppo_agent.get_action_and_value(mb_obs, mb_actions.long() if mb_actions.ndim>0 else mb_actions)
                # newvalue may be single or tuple; ensure shape
                if isinstance(newvalue, tuple) or (hasattr(newvalue, 'shape') and newvalue.ndim>1 and newvalue.shape[1] > 1):
                    # If network returns (ext, int), take ext for value loss here
                    if isinstance(newvalue, tuple):
                        new_ext_value = newvalue[0].view(-1)
                        new_int_value = newvalue[1].view(-1)
                    else:
                        # fallback
                        new_ext_value = newvalue.view(-1)
                        new_int_value = newvalue.view(-1)
                else:
                    new_ext_value = newvalue.view(-1)
                    new_int_value = newvalue.view(-1)

                # policy ratio
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                # approx_kl for diagnostics
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                # normalize advantages if requested (common)
                if self.ppo_agent.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (clipped surrogate) - COMMON PPO
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped) - COMMON PPO
                if self.ppo_agent.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_value - mb_ext_returns) ** 2
                    ext_v_clipped = mb_ext_values + torch.clamp(new_ext_value - mb_ext_values, -clip_coef, clip_coef)
                    ext_v_loss_clipped = (ext_v_clipped - mb_ext_returns) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_value - mb_ext_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                # RND forward loss for this minibatch
                # prepare rnd minibatch inputs from precomputed rnd_next_obs
                mb_rnd = rnd_next_obs = None  # placeholder in case user wants per-minibatch RND from b_obs
                # Instead, compute forward_loss from rnd inputs: we already computed rnd inputs per step earlier (rnd_inputs_stack)
                # For simplicity, use rnd_inputs_stack flattened to match b_inds ordering
                rnd_flat = torch.from_numpy(rnd_inputs_stack).to(device).float()  # shape (batch_size, 1, 84,84)
                mb_rnd = rnd_flat[mb_inds]

                # compute predictor vs target
                predict_feat, target_feat = self.rnd(mb_rnd)
                fwd_loss = F.mse_loss(predict_feat, target_feat.detach(), reduction='none').mean(-1)

                # mask a proportion of samples to update predictor only on subset
                mask = torch.rand(len(fwd_loss), device=device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (fwd_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1.0], device=device))

                # full loss: policy + value + rnd forward loss
                loss = pg_loss - self.ppo_agent.ent_coef * entropy_loss + ext_v_loss * self.ppo_agent.vf_coef + forward_loss

                # gradient step on combined parameters (ppo + rnd.predictor)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.ppo_agent.parameters()) + list(self.rnd.predictor.parameters()), self.ppo_agent.max_grad_norm)
                self.optimizer.step()

                # keep last values for reporting
                ppo_loss_val = loss.item()
                rnd_forward_loss = forward_loss.item()
                
                
                self.variance += rnd_forward_loss

        return ppo_loss_val, rnd_forward_loss
    

class Random_Exploration_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.num_actions = envs.single_action_space.n

    def get_action(self, state):
        return np.random.choice(self.num_actions, size = self.envs.num_envs)
    def train(self, obs, actions, rewards, logprobs, dones, values, next_obs, next_done):
        return 0
    
    def reduce_lr(self, iteration):
        return

class Two_Modes_ADEU(nn.Module):


    def __init__(self,
                 envs,
                 num_steps,
                 gamma,
                 batch_size,
                 minibatch_size,
                 gae_lambda,
                 update_epochs,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 learning_rate,
                 num_iterations,
                 anneal_lr,
                 device,
                 head_start,
                 starting_mode,
                 maximum_timesteps,
                 **kwargs):
        super().__init__()
        self.exploitation_agent = PPO_Agent(envs,
                                            num_steps,
                                            gamma,
                                            batch_size,
                                            minibatch_size,
                                            gae_lambda,
                                            update_epochs,
                                            ent_coef,
                                            vf_coef,
                                            max_grad_norm,
                                            learning_rate,
                                            num_iterations,
                                            anneal_lr,
                                            device,
                                            head_start,
                                            )
        
        self.exploration_agent = Random_Exploration_Agent(envs)
        self.n_envs = envs.num_envs
        self.current_timestep = -1
        self.maximum_timesteps = maximum_timesteps

        self.k = 5
        self.past_values = np.zeros([maximum_timesteps, self.n_envs] )
        self.past_rewards = np.zeros([maximum_timesteps, self.n_envs])
        self.previous_d_promises = np.zeros([maximum_timesteps, self.n_envs])
        self.gamma = gamma
        self.homeostasis_threshold = np.zeros([maximum_timesteps,self.n_envs]) # we inityiallize it to 0 since we would assume that the uncertainty for the first ever episode is high and we would like the threshold to be zero in order to explore
        self.homeostasis_rho = 0.01
        self.uncertainty = 0
        self.episodes_with_no_evaluation = 1
        self.starting_mode = starting_mode
        self.device = device
        

    def reset (self, ):
        self.current_timestep = -1
        self.past_values = np.zeros([self.maximum_timesteps, self.n_envs] )
        self.past_rewards = np.zeros([self.maximum_timesteps, self.n_envs])
        self.homeostasis_threshold = self.homeostasis()
        self.episodes_with_no_evaluation+=1
    
    def reduce_lr(self, iteration):
        self.exploitation_agent.reduce_lr(iteration)
        self.exploration_agent.reduce_lr(iteration)
    
    def get_clear_action(self, state):
        return self.exploitation_agent.get_clear_action(state, evaluate = True)

    def get_variance(self, ):
        retval = self.uncertainty/self.episodes_with_no_evaluation
        self.episodes_with_no_evaluation = 0
        return retval, 0
    
    def update_values_rewards(self, values, rewards):
        self.past_values[self.current_timestep, :] = values.detach().cpu().numpy()
        self.past_rewards[self.current_timestep, :] = rewards
        return



    def get_action_and_value(self, state):
        self.current_timestep+=1
        exploitation_action, log_prob, entropy, exploiation_value = self.exploitation_agent.get_action_and_value(x = state)
        uncertainty = self.calculate_uncertainty(exploiation_value, ) #uncertainty is (num_envs, 1)

        

        self.uncertainty += np.mean(uncertainty)
        self.previous_d_promises[self.current_timestep,:] = uncertainty[:]

        actions = []
        log_probs = []
        entropys = []
        exploiation_values = []

        for i in range(self.n_envs):

            if uncertainty[i] < self.homeostasis_threshold[self.current_timestep, i]:
                #uncertainty is low, exploit
                actions.append(exploitation_action[i])
                log_probs.append(log_prob[i])
                entropys.append(entropy[i])
                exploiation_values.append(exploiation_value[i])
            else:
                #the action in the next couple of lines is THE SAME ONE (but one is in torch)
                action = self.exploration_agent.get_action(state)
                action , log_prob, entropy, exploration_value = self.exploitation_agent.get_action_and_value(state, action = torch.from_numpy(action).to(self.device))
                
                actions.append(action[i])
                log_probs.append(log_prob[i])
                entropys.append(entropy[i])
                exploiation_values.append(exploration_value[i])
        actions = torch.Tensor(actions).to(self.device)
        actions = torch.round(actions, decimals=0).int()

        return actions, torch.Tensor(log_probs).to(self.device), torch.Tensor(entropys).to(self.device), torch.Tensor(exploiation_values).to(self.device), 

    
    def calculate_uncertainty(self, value:torch.Tensor):

        if self.current_timestep < self.k:
            if self.starting_mode == 'exploration' :
                return np.ones(self.n_envs)
            return np.zeros(self.n_envs)

        value = value.detach().cpu().numpy()
        v_t_minus_k = self.past_values[self.current_timestep - self.k] if (self.current_timestep - self.k) >= 0 \
                                                                       else 0
        r_t_minus_k = self.past_rewards[self.current_timestep - self.k : self.current_timestep ] if (self.current_timestep - self.k) >= 0 \
                                                                       else 0
        s = np.zeros(self.n_envs)
        for i in range(self.n_envs):
            s[i] = (np.sum(np.array([self.gamma **i *r_t_minus_k[max(-i-1, -5)]  for i in range(r_t_minus_k.size)])))

        D_promise = v_t_minus_k - s - (self.gamma**self.k)* np.squeeze(value)
        D_promise = np.abs(D_promise)
        return D_promise
    
    def homeostasis(self, ):

        rho = self.homeostasis_rho
        x_values = self.previous_d_promises
        T = len(x_values)

        # Initialize variables
        x_bar = 0.0
        x2_bar = 1.0
        x_plus_bar = 1.0

        y = np.zeros([T, self.n_envs])

        for t in range(T):
            x_t = x_values[t]
            tau = min(t + 1, 100 / rho)  # t starts at 0, so use (t+1)

            # Update moving average and variance
            x_bar = (1 - 1 / tau) * x_bar + (1 / tau) * x_t
            x2_bar = (1 - 1 / tau) * x2_bar + (1 / tau) * (x_t - x_bar) ** 2

            # Standardize and exponentiate
            x_plus = np.exp((x_t - x_bar) / np.sqrt(x2_bar + 1e-8))  # small epsilon for stability

            # Update transformed moving average
            x_plus_bar = (1 - 1 / tau) * x_plus_bar + (1 / tau) * x_plus

            # Sample from Bernoulli
            p = np.minimum(np.ones(self.n_envs), rho * x_plus / (x_plus_bar + 1e-8))
            y[t, :] = np.random.binomial(1, p)

        return y
    
    def train(self, obs, actions, rewards, logprobs, dones, values, next_obs, next_done):
        loss = 0
        loss += self.exploitation_agent.train(obs, actions, rewards, logprobs, dones, values, next_obs, next_done)
        loss += self.exploration_agent.train(obs, actions, rewards, logprobs, dones, values, next_obs, next_done)
        return loss