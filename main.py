import comet_ml
import tianshou as ts
import gym
from tianshou.utils.net.common import Net
import tianshou.utils.net.discrete as discrete
import torch
from tianshou.utils import WandbLogger, TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

POLICY_CONFIG = {
    "is_double": False,
    "discount_factor": 0.99,
    "target_update_freq": 100,
}

def train():
    optim_config = {
        "lr": 0.001
    }

    env = gym.make("LunarLander-v2")
    train_env = ts.env.DummyVectorEnv([lambda: gym.make("LunarLander-v2") for _ in range(10)])
    test_env = ts.env.DummyVectorEnv([lambda: gym.make("LunarLander-v2") for _ in range(2)])

    state_shape = env.observation_space.shape or env.observation_space.n
    actions_n = env.action_space.shape or env.action_space.n

    actor = Net(state_shape, actions_n, hidden_sizes=[512, 256])
    actor_optim = torch.optim.AdamW(actor.parameters(), **optim_config)

    dqn = ts.policy.DQNPolicy(
        actor, actor_optim,
        **POLICY_CONFIG
    )

    train_collector = ts.data.Collector(dqn, train_env, ts.data.VectorReplayBuffer(500000, 10), exploration_noise=False)
    test_collector = ts.data.Collector(dqn, test_env, exploration_noise=False)

    # Comet
    experiment = comet_ml.Experiment(api_key="sB2qT71Uklji1EGNlkZ2WhuzL")
    experiment.set_name(f"{dqn.__class__.__name__}_LunarLander-v2")
    experiment.log_parameters({**POLICY_CONFIG, **optim_config})
    experiment.set_model_graph(dqn)
    # Tensorboard
    writer = SummaryWriter(f"./logs/lunar-lander/")
    writer.add_text("hparams", str({**POLICY_CONFIG, **optim_config}))
    logger = TensorboardLogger(writer, train_interval=10, update_interval=10, test_interval=1)
    # Training
    result = ts.trainer.offpolicy_trainer(
        dqn, train_collector, test_collector,
        max_epoch=1000,
        batch_size=128,
        step_per_epoch=1000, step_per_collect=10, update_per_step=1,
        episode_per_test=5,
        train_fn=lambda epoch, step: dqn.set_eps(1.0 * (1.0 - epoch/1000)),
        stop_fn=lambda rew: rew >= env.spec.reward_threshold,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), f"checkpoints/dqn.pth"),
        logger=logger
    )
    experiment.log_model("Best model", "checkpoints/dqn.pth")

    train_env.close()
    test_env.close()
    env.close()

def test():
    env = gym.make("LunarLander-v2", render_mode="human")

    state_shape = env.observation_space.shape or env.observation_space.n
    actions_n = env.action_space.shape or env.action_space.n
    env = ts.env.DummyVectorEnv([lambda : env])
    env.reset()

    actor = Net(state_shape,  actions_n, hidden_sizes=[512, 256])
    optim = torch.optim.AdamW(actor.parameters(), lr=0.001)
    policy = ts.policy.DQNPolicy(actor, optim, **POLICY_CONFIG)
    policy.load_state_dict(torch.load("checkpoints/dqn.pth"))
    # Testing
    policy.eval()
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=10, render=1/60)
    env.close()

if __name__ == "__main__":
    test()