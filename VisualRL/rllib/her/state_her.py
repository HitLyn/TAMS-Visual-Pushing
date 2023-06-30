import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
import multiprocessing as mp

import numpy as np

from VisualRL.common.utils import get_device, set_seed_everywhere
from VisualRL.rllib.common.torch_layers import make_feature_extractor
from VisualRL.rllib.her.sac_policy import SACPolicy
from VisualRL.rllib.her.state_her_replay_buffer import SHerReplayBuffer as HerReplayBuffer
from VisualRL.rllib.her.her_replay_buffer_test import HerReplayBufferTest
from VisualRL.rllib.common.utils import polyak_update

from robogym.envs.push.push_a3 import make_env

ACTION_SCALE = 0.7
class S_HER:
    def __init__(
            self,
            observation_space,
            action_space,
            goal_space,
            env,
            feature_dims,
            min_action,
            max_action,
            max_episode_steps,
            train_freq,
            train_cycle,
            net_class = "Flatten",
            target_update_interval = 20,
            save_interval = 100,
            gradient_steps = 50,
            learning_rate = 1e-3,
            buffer_size = 1e6,
            learning_starts = 100,
            batch_size = 256,
            tau = 0.005,
            gamma = 0.99,
            device = None,
            seed = 1,
            relative_goal = True,
            goal_type = 'pos',
            test = False,
            ):

        self.observation_space = observation_space # network size
        self.action_space = action_space
        self.goal_space = goal_space
        self.env = env
        self.buffer_obs_size = observation_space - goal_space # buffer obs shape = 9
        self.max_episode_steps = max_episode_steps
        self.feature_dims = observation_space if net_class == "Flatten" else feature_dims
        self.net_class = net_class
        self.learning_rate = learning_rate
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.min_action = min_action
        self.max_action = max_action
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.seed = seed
        self.relative_goal = relative_goal
        self.goal_type = goal_type
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.train_cycle = train_cycle
        self.num_workers = 5

        self.dims = self.get_dims()

        self._episode_num = 0
        self.num_timesteps = 0
        self._n_updates = 0


        if test:
            self.rollout_buffer = HerReplayBufferTest(
                buffer_size,
                max_episode_steps,
                self.buffer_obs_size,
                goal_space,
                action_space,
                device,
                relative_goal = self.relative_goal,
            )
        else:
            self.rollout_buffer = HerReplayBuffer(
                    buffer_size,
                    max_episode_steps,
                    self.buffer_obs_size,
                    goal_space,
                    action_space,
                    device,
                    relative_goal = self.relative_goal,
                    goal_type = self.goal_type,
                    )

        self.feature_extractor = make_feature_extractor(
                net_class,
                observation_space = observation_space,
                feature_dims = feature_dims,
                device = device
                )

        self.policy = SACPolicy(
                observation_space,
                action_space,
                self.feature_dims,
                self.feature_extractor,
                self.rollout_buffer,
                device,
                min_action,
                max_action,
                learning_rate = learning_rate,
                net_class = self.net_class,
                )

        self.policy.to(self.device)

        # entropy item
        self.target_entropy = -np.prod(self.action_space).astype(np.float32)
        self.log_ent_coef = torch.log(torch.ones(1, device = self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr = 3e-4)
        self.ent_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.ent_coef_optimizer, 0.999)

        # actor item
        self.actor = self.policy.actor
        self.actor_scheduler = self.policy.actor_scheduler
        # critic item
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.critic_scheduler = self.policy.critic_scheduler

        # prepare to learn
        self._setup_learn()

    def _setup_learn(self):
        self._episode_num = 0
        self.num_collected_episodes = 0

    def get_dims(self):
        self.dims = dict()
        self.dims['observation_space'] = self.observation_space
        self.dims['buffer_obs_size'] = self.buffer_obs_size
        self.dims['action'] = self.action_space
        self.dims['goal'] = self.goal_space
        return self.dims

    def _select_action(self, observation, achieved_goal, desired_goal):
        observation = observation.reshape(-1, self.dims['buffer_obs_size'])
        # compute relative goal
        if self.relative_goal:
            desired_goal = (desired_goal - achieved_goal).reshape(-1, self.dims['goal'])
        else:
            desired_goal = desired_goal.reshape(-1, self.dims['goal'])
        obs_input = np.concatenate([observation, desired_goal], axis = 1)
        obs_input = torch.as_tensor(obs_input).float().to(self.device)
        scaled_action = self.policy.predict(obs_input, determinstic = True).squeeze().cpu().numpy()
        return scaled_action

    def _sample_action(self, observation, achieved_goal, desired_goal, cpu = False):
        # reshape and normalize observations for network
        observation = observation.reshape(-1, self.dims['buffer_obs_size'])
        if self.relative_goal:
            desired_goal = (desired_goal - achieved_goal).reshape(-1, self.dims['goal'])
        else:
            desired_goal = desired_goal.reshape(-1, self.dims['goal'])
        obs_input = np.concatenate([observation, desired_goal], axis=1)
        obs_input = torch.as_tensor(obs_input).float().to(torch.device("cpu")) if cpu else torch.as_tensor(obs_input).float().to(self.device)

        scaled_action = self.policy.predict(obs_input, determinstic = False).squeeze().cpu().numpy()
        return scaled_action

    def collect_rollouts(self, env, writer):
        success_stats = []
        reward_stats = []
        episode = 0
        print(f"collecting rollouts: {self.num_collected_episodes}")
        while episode < self.train_freq:
            obs_dict = env.reset()
            observation = np.empty(self.dims['buffer_obs_size'], np.float32)
            achieved_goal = np.empty(self.dims['goal'], np.float32)
            desired_goal = np.empty(self.dims['goal'], np.float32)
            observation[:] = obs_dict['observation']
            achieved_goal[:] = obs_dict['achieved_goal']
            desired_goal[:] = obs_dict['desired_goal']

            obs, a_goals, acts, d_goals, successes, dones, rewards = [], [], [], [], [], [], []
            with torch.no_grad():
                for t in range(self.max_episode_steps):
                    observation_new = np.empty(self.dims['buffer_obs_size'], np.float32)
                    achieved_goal_new = np.empty(self.dims['goal'], np.float32)
                    # success = np.zeros(1)

                    # step env
                    action= self._sample_action(observation, achieved_goal, desired_goal) # action is squashed to [-1, 1] by tanh function
                    obs_dict_new, reward, done, _ = env.step(ACTION_SCALE*action)
                    observation_new[:] = obs_dict_new['observation']
                    achieved_goal_new[:] = obs_dict_new['achieved_goal']
                    success = np.array(obs_dict_new['is_success'])

                    # store transitions
                    dones.append(done)
                    obs.append(observation.copy())
                    a_goals.append(achieved_goal.copy())
                    acts.append(action.copy())
                    d_goals.append(desired_goal.copy())
                    successes.append(success.copy())
                    rewards.append(reward)

                    # update states
                    observation[:] = observation_new.copy()
                    achieved_goal[:] = achieved_goal_new.copy()

            obs.append(observation.copy())
            a_goals.append(achieved_goal.copy())

            episode_transition = dict(
                o = np.array(obs).copy(),
                u = np.array(acts).copy(),
                g = np.array(d_goals).copy(),
                ag = np.array(a_goals).copy(),
                r = np.array(rewards).copy())
            # stats
            episode += 1
            self.num_collected_episodes += 1
            success_stats.append(successes[-1])
            reward_stats.append(np.mean(rewards))
            # add transition to replay buffer
            self.rollout_buffer.add_episode_transitions(episode_transition)

        success_rate = np.mean(np.array(success_stats))
        mean_reward = np.mean(np.array(reward_stats))
        #TODO write success_rate to logger here
        # writer.add_scalar("train/success_rate", success_rate, self._n_updates)
        writer.add_scalar("train/mean_reward", mean_reward, self._n_updates)

    def mp_collect_rollouts(self, i, seed_list, mp_list, policy, writer):
        env = make_env()
        set_seed_everywhere(seed_list[i])
        # stats
        success_stats = []
        # rollout for all workers
        obs_dict = env.reset()
        observation = np.empty(self.dims['buffer_obs_size'], np.float32)
        achieved_goal = np.empty(self.dims['goal'], np.float32)
        desired_goal = np.empty(self.dims['goal'], np.float32)
        observation[:] = obs_dict['observation']
        achieved_goal[:] = obs_dict['achieved_goal']
        desired_goal[:] = obs_dict['desired_goal']

        obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(self.max_episode_steps):
                observation_new = np.empty(self.dims['buffer_obs_size'], np.float32)
                achieved_goal_new = np.empty(self.dims['goal'], np.float32)
                # success = np.zeros(1)

                # step env
                action= self._sample_action(observation, achieved_goal, desired_goal, cpu = True) # action is squashed to [-1, 1] by tanh function
                obs_dict_new, reward, done, _ = env.step(ACTION_SCALE*action)
                observation_new[:] = obs_dict_new['observation']
                achieved_goal_new[:] = obs_dict_new['achieved_goal']
                success = np.array(obs_dict_new['is_success'])

                # store transitions
                dones.append(done)
                obs.append(observation.copy())
                a_goals.append(achieved_goal.copy())
                acts.append(action.copy())
                d_goals.append(desired_goal.copy())
                successes.append(success.copy())

                # update states
                observation[:] = observation_new.copy()
                achieved_goal[:] = achieved_goal_new.copy()

        obs.append(observation.copy())
        a_goals.append(achieved_goal.copy())

        episode_transition = dict(
            o = np.array(obs).copy(),
            u = np.array(acts).copy(),
            g = np.array(d_goals).copy(),
            ag = np.array(a_goals).copy())

        # add transition to mp_list
        mp_list.append(episode_transition)


    def to(self, device):
        self.device = device
        self.policy.to(device)

    def learn(self, total_episodes, eval_freq, num_eval_episodes, writer, model_path, multiprocess = False):
        # rollout and train model in turn
        while self.num_collected_episodes < total_episodes:
            if multiprocess:
                # move policy to cpu
                self.policy.to(torch.device("cpu"))
                # print('create mp target')
                with torch.no_grad():
                    tmp_seed_list = np.random.randint(1, 10000, size=self.num_workers)
                    mp_list = mp.Manager().list()
                    workers = [mp.Process(target=self.mp_collect_rollouts,
                                          args=(i, tmp_seed_list, mp_list, self.policy, writer))
                               for i in range(self.num_workers)]
                    # embed()
                    [worker.start() for worker in workers]
                    [worker.join() for worker in workers]
                    mp_list = list(mp_list)
                    # pool = mp.Pool(self.num_workers)
                    # mp_list = [pool.apply(self.mp_collect_rollouts_, args = (i, tmp_seed_list, env)) for i in range(self.num_workers)]
                self.rollout_buffer.add_episode_transitions_list(mp_list)
                self.num_collected_episodes += 1
                print(f"collecting rollouts with {self.num_workers} workers, episodes {self.num_collected_episodes}")
            else:
                self.collect_rollouts(self.env, writer)
            if self.num_collected_episodes >= self.learning_starts:
                # move policy back to gpu
                # self.policy.to(self.device)
                self.train(self.gradient_steps, self.batch_size, writer)
                if self.num_collected_episodes % eval_freq == 0:
                    self.eval(self.env, num_eval_episodes, writer)
                    # save
                if self.num_collected_episodes % self.save_interval == 0:
                    self.save(model_path, self.num_collected_episodes)


    def eval(self, env, num_eval_episodes, writer):
        print(f"evaluate after {self.num_collected_episodes} episodes")
        reward_stats, success_rate_stats= [], []
        for episode in range(num_eval_episodes):
            obs_dict = env.reset()
            rewards = []
            with torch.no_grad():
                for step in range(self.max_episode_steps):
                    action = self._select_action(obs_dict['observation'], obs_dict['achieved_goal'], obs_dict['desired_goal'])
                    obs_dict, reward, done, _ = env.step(ACTION_SCALE*action)
                    rewards.append(reward)
            success = obs_dict['is_success']
            reward_stats.append(np.mean(np.array(rewards)))
            success_rate_stats.append(success)

        mean_reward = np.mean(np.array(reward_stats))
        mean_success_rate = np.mean(np.array(success_rate_stats))
        # TODO write stats to logger here
        writer.add_scalar("eval/mean_reward", mean_reward, self._n_updates)
        writer.add_scalar("eval/success_rate", mean_success_rate, self._n_updates)
        print(f"success_rate: {mean_success_rate}")

    def train(self, gradient_steps, batch_size, writer):
        # optimizers
        # train
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.rollout_buffer.sample(batch_size)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data["goal_obs_con"])
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef*(log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())
            # optimize entropy coefficient
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            with torch.no_grad():
                # select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data["next_goal_obs_con"])
                # compute next Q values
                next_q_values = torch.cat(self.critic_target(replay_data["next_goal_obs_con"], next_actions), dim = 1)
                next_q_values, _ = torch.min(next_q_values, dim = 1, keepdim = True)
                # add entropy term
                next_q_values = next_q_values - ent_coef*next_log_prob.reshape(-1, 1)
                # td error, entropy term
                target_q_values = replay_data["rewards"] + (1 - replay_data["dones"]) * self.gamma * next_q_values

            # get current Q values estimates for each critic net
            current_q_values = self.critic(replay_data["goal_obs_con"], replay_data["actions"])
            # compute critic loss
            # embed();exit()
            critic_loss = 0.5*sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())
            # optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # compute actor loss
            q_values_pi = torch.cat(self.critic(replay_data["goal_obs_con"], actions_pi), dim = 1)
            min_qf_pi, _ = torch.min(q_values_pi, dim = 1, keepdim = True)
            actor_loss = (ent_coef*log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            # optimize actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # updata target network
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # if gradient_step % self.target_update_interval == 0:
        # update critic target
        # polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        # update learning rate
        schedulers = [self.actor_scheduler, self.critic_scheduler, self.ent_scheduler]
        # self._update_learning_rate(schedulers)
        # TODO write summary to logger here
        writer.add_scalar("train/ent_coef", np.mean(ent_coefs), self._n_updates)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), self._n_updates)
        writer.add_scalar("train/critic_loss", np.mean(critic_losses), self._n_updates)
        if len(ent_coef_losses) > 0:
            writer.add_scalar("train/ent_coef_loss", np.mean(ent_coef_losses), self._n_updates)


    def _update_learning_rate(self, schedulers):
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def save(self, path, step):
        torch.save(self.policy.state_dict(), "%s/her_%s.pt" % (path, step))

    def load(self, path, step, map_location = None):
        self.policy.load_state_dict(torch.load("%s/her_%s.pt" % (path, step), map_location = map_location))