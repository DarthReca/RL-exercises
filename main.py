import tianshou as ts
import gym
from tianshou.utils.net.common import Net
import tianshou.utils.net.discrete as discrete
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import WandbLogger


logger = WandbLogger()
writer = SummaryWriter("logs/lunar-lander")
logger.load(writer)

env = gym.make("LunarLander-v2")
train_env = ts.env.DummyVectorEnv([lambda: gym.make("LunarLander-v2") for _ in range(10)])
test_env = ts.env.DummyVectorEnv([lambda: gym.make("LunarLander-v2") for _ in range(2)])

state_shape = env.observation_space.shape or env.observation_space.n
actions_n = env.action_space.shape or env.action_space.n

actor = discrete.Actor(Net(state_shape, hidden_sizes=[32, 32]), action_shape=actions_n)
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)

critic_1 = discrete.Critic(Net(state_shape, actions_n, hidden_sizes=[32, 32]), last_size=actions_n)
critic_1_optim = torch.optim.Adam(critic_1.parameters(), lr=3e-4)

critic_2 = discrete.Critic(Net(state_shape, actions_n, hidden_sizes=[32, 32]), last_size=actions_n)
critic_2_optim = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

dqn = ts.policy.DiscreteSACPolicy(
    actor, actor_optim,
    critic_1, critic_1_optim,
    critic_2, critic_2_optim,
    tau=0.005, gamma=0.9,
)

train_collector = ts.data.Collector(dqn, train_env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(dqn, test_env, exploration_noise=True)


def train_fn(epoch, step):
    pass


ts.trainer.offpolicy_trainer(
    dqn, train_collector, test_collector, max_epoch=10, batch_size=128,
    step_per_epoch=10000, step_per_collect=10,
    train_fn=train_fn,
    episode_per_test=10, stop_fn=lambda rew: rew >= env.spec.reward_threshold,
    logger=logger
)

train_env.close()
test_env.close()
