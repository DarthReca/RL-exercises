import comet_ml
import tianshou as ts
import gym
from tianshou.utils.net.common import Net
import tianshou.utils.net.discrete as discrete
import torch
from tianshou.utils import TensorboardLogger, MovAvg
from torch.utils.tensorboard import SummaryWriter
import models
from math import log

POLICY_CONFIG = {"gamma": 0.9, "tau": 0.05, "alpha": 0.2}


def train():
    optim_config = {"lr": 0.0003}

    env = gym.make("SpaceInvaders-v4")
    train_env = ts.env.SubprocVectorEnv(
        [lambda: gym.make("SpaceInvaders-v4") for _ in range(1)]
    )
    test_env = ts.env.DummyVectorEnv(
        [lambda: gym.make("SpaceInvaders-v4") for _ in range(1)]
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    actions_n = env.action_space.shape or env.action_space.n

    starting_alpha = torch.zeros(1, requires_grad=True)
    POLICY_CONFIG["alpha"] = (
        -0.98 * log(1 / actions_n),
        starting_alpha,
        torch.optim.AdamW([starting_alpha], **optim_config),
    )

    actor = models.PolicyNetwork(state_shape, actions_n)
    actor_optim = torch.optim.AdamW(actor.parameters(), **optim_config)

    critic_1 = models.QValueNetwork(state_shape, actions_n)
    critic_1_optim = torch.optim.AdamW(actor.parameters(), **optim_config)

    critic_2 = models.QValueNetwork(state_shape, actions_n)
    critic_2_optim = torch.optim.AdamW(actor.parameters(), **optim_config)

    dqn = ts.policy.DiscreteSACPolicy(
        actor,
        actor_optim,
        critic_1,
        critic_1_optim,
        critic_2,
        critic_2_optim,
        **POLICY_CONFIG,
    )

    train_collector = ts.data.Collector(
        dqn,
        train_env,
        ts.data.VectorReplayBuffer(500000, len(train_env)),
        exploration_noise=False,
    )
    test_collector = ts.data.Collector(dqn, test_env, exploration_noise=False)

    # Comet
    experiment = comet_ml.Experiment(api_key="sB2qT71Uklji1EGNlkZ2WhuzL")
    experiment.set_name(f"{dqn.__class__.__name__}_LunarLander-v2")
    experiment.log_parameters({**POLICY_CONFIG, **optim_config})
    experiment.set_model_graph(dqn)
    # Tensorboard
    writer = SummaryWriter(f"./logs/lunar-lander/")
    writer.add_text("hparams", str({**POLICY_CONFIG, **optim_config}))
    logger = TensorboardLogger(
        writer, train_interval=10, update_interval=10, test_interval=1
    )
    # Training
    result = ts.trainer.offpolicy_trainer(
        dqn,
        train_collector,
        test_collector,
        max_epoch=100,
        batch_size=64,
        step_per_epoch=1000,
        step_per_collect=4,
        update_per_step=1,
        episode_per_test=10,
        save_best_fn=lambda policy: torch.save(
            policy.state_dict(), f"checkpoints/sac.pth"
        ),
        logger=logger,
    )
    experiment.log_model("Best model", "checkpoints/sac.pth")

    train_env.close()
    test_env.close()
    env.close()


def test():
    env = gym.make("LunarLander-v2", render_mode="human")

    state_shape = env.observation_space.shape or env.observation_space.n
    actions_n = env.action_space.shape or env.action_space.n
    env = ts.env.DummyVectorEnv([lambda: env])
    env.reset()

    actor = Net(state_shape, actions_n, hidden_sizes=[512, 256])
    optim = torch.optim.AdamW(actor.parameters(), lr=0.001)
    policy = ts.policy.DQNPolicy(actor, optim, **POLICY_CONFIG)
    policy.load_state_dict(torch.load("checkpoints/dqn.pth"))
    # Testing
    policy.eval()
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=5, render=1 / 60)
    env.close()


if __name__ == "__main__":
    train()
