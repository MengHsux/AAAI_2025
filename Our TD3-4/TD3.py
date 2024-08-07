import copy
import numpy as np
import torch
import torch.nn.functional as F
from modules_TD3 import MLPActor, MLPCritic
from TD3UU.utils import ReplayBuffer, get_W
from torch.utils.tensorboard import SummaryWriter
from SpikeActor import SpikeActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(
            self,
            args,
            writer,
            ac_kwargs=dict()
    ):
        # RL hyperparameters
        self.max_action = args.max_action
        self.discount = args.discount
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

        # Neural networks
        self.actor = SpikeActor(args.state_dim, args.action_dim, args.max_action, **ac_kwargs).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic1 = MLPCritic(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)

        self.critic2 = MLPCritic(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.critic_target2 = copy.deepcopy(self.critic2)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.total_it = 0
        self.nstep = args.nstep
        self.delay_nstep = args.delay_nstep
        self.writer: SummaryWriter = writer
        self.tb_interval = int(args.T_end / 1000)

    def select_action(self, state) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, batch_size=256):
        self.total_it += 1

        # Delay to use multi-step TD target
        current_nstep = self.nstep if self.total_it >= self.delay_nstep else 1

        if self.total_it % self.tb_interval == 0: self.writer.add_scalar('current_nstep', current_nstep, self.total_it)

        state, action, next_state, reward, not_done, _, reset_flag = replay_buffer.sample(batch_size, current_nstep)

        with torch.no_grad():
            noise = (
                    torch.randn_like(action[:, 0]) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            accum_reward = torch.zeros(reward[:, 0].shape).to(device)
            have_not_done = torch.ones(not_done[:, 0].shape).to(device)
            have_not_reset = torch.ones(not_done[:, 0].shape).to(device)
            modified_n = torch.zeros(not_done[:, 0].shape).to(device)
            for k in range(current_nstep):
                accum_reward += have_not_reset * have_not_done * self.discount ** k * reward[:, k]
                have_not_done *= torch.maximum(not_done[:, k], 1 - have_not_reset)
                if k == current_nstep - 1:
                    break
                have_not_reset *= (1 - reset_flag[:, k])
                modified_n += have_not_reset
            modified_n = modified_n.type(torch.long)
            nstep_next_state = next_state[np.arange(batch_size), modified_n[:, 0]]
            next_action = (
                    self.actor_target(nstep_next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1 = self.critic_target1(nstep_next_state, next_action)
            target_Q2 = self.critic_target2(nstep_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if current_nstep == 1:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(
                    target_Q.shape) * self.discount * target_Q
            else:
                target_Q = accum_reward.reshape(target_Q.shape) + have_not_done.reshape(
                    target_Q.shape) * self.discount ** (modified_n + 1) * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(state[:, 0], action[:, 0])
        current_Q2 = self.critic2(state[:, 0], action[:, 0])
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if self.total_it % self.tb_interval == 0: self.writer.add_scalar('critic_loss', critic_loss.item(),
                                                                         self.total_it)

        # Optimize the critic
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss: torch.Tensor = -self.critic1(state[:, 0], self.actor(state[:, 0])).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
