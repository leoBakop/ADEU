import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from network import PPONetwork


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
    

    def get_clear_action(self, x):
        if self.training_steps < self.head_start:
            return np.random.choice(self.num_actions)
        print('I should check if this return just a single action')
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self.network(x, return_actor = True)
        return np.argmax(logits.cpu().detach().numpy())
    

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


