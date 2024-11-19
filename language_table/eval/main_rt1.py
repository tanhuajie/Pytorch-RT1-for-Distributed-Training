import imageio
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
parent_dir = os.path.dirname(current_dir) # 上一级目录
# parent_dir = os.path.dirname(parent_dir1)
sys.path.append(parent_dir) 
# from tf_agents.specs import tensor_spec
# import tensorflow as tf
import collections
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np
import torch
# from distribute_train_tf import get_args,create_model,create_train_dataset
# from language_table.common import rt1_tokenizer
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers
from language_table.train import policy as jax_policy
# from ml_collections import config_flags

# import tensorflow as tf
# import tensorflow_hub as hub
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

from distribute_train import RT1_Lightning

from pytorch_robotics_transformer.tokenizers.utils import batched_space_sampler
from pytorch_robotics_transformer.tokenizers.utils import np_to_tensor

_WORKDIR = flags.DEFINE_string("workdir","/gemini/data-2/test/pytorch-RT1/eval", "working dir")

def get_ckpt_model(ckpt_path):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--device", type = str, default = "gpu")
    parser.add_argument("--gpus", type = str, default = "0,1,2,3")
    parser.add_argument("--max_epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--num_workers", type = int, default = 15)
    parser.add_argument('--milestones', type=int, nargs='+', default = [50, 75, 90])
    parser.add_argument('--lr', type=float, default = 5e-4)

    parser.add_argument("--exp_name", type = str, default = "exp_rt1")
    parser.add_argument("--log_dir", type = str, default = "./exp/logs")
    parser.add_argument("--ckpt_dir", type = str, default = "./exp/ckpt")
    parser.add_argument("--log_every_n_steps", type = int, default = 500)
    parser.add_argument("--ckpt_every_n_epochs", type = int, default = 1)

    parser.add_argument("--dataset_dir", type = str, default = "/gemini/data-2/data/pytorch_b2b_sim_np")
    parser.add_argument("--train_episode", type = int, default = 7800)
    parser.add_argument("--test_episode", type = int, default = 50)
    parser.add_argument("--eval_episode", type = int, default = 50)

    parser.add_argument('--random_crop_factor', type=float, default = 0.95)
    parser.add_argument("--height", type = int, default = 256)
    parser.add_argument("--width", type = int, default = 456)
    parser.add_argument("--seq_len", type = int, default = 6)

    args = parser.parse_args()

    print('Loading CKPT ...')
    model = RT1_Lightning.load_from_checkpoint(ckpt_path, args=args)
    print('FINISH CKPT ...')
    network_state = batched_space_sampler(model.model._state_space, batch_size=1)
    # network_state = model.model._state_space.sample()
    network_state = np_to_tensor(network_state, 'cuda')
    model.eval().to('cuda')
    return model, network_state

# def get_ckpt_model():
#     time_sequence_length = 6  # 常量，来自论文每次预测使用6张图片
#     args = get_args()
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     with tf.device('/gpu:0'):
#         network = create_model(args)
#         network_state = tensor_spec.sample_spec_nest(
#             network.state_spec, outer_dims=[1])
#         network_state['context_image_tokens'] = tf.zeros_like(network_state['context_image_tokens'])
#         network_state['action_tokens'] = tf.zeros_like(network_state['action_tokens'])
#         network_state['seq_idx'] = tf.zeros_like(network_state['seq_idx'])
#         ckpt = tf.train.Checkpoint(step=tf.Variable(9),model=network)
#         # if tf.train.latest_checkpoint(args.loaded_checkpoints_dir):
#         ckpt.restore(tf.train.latest_checkpoint(args.loaded_checkpoints_dir)).expect_partial()
#         print("从 %s 恢复模型" % (tf.train.latest_checkpoint(args.loaded_checkpoints_dir)))
#         return ckpt.model, network_state

def evaluate_checkpoint(workdir, config, ckpt):
  """Evaluates the given checkpoint and writes results to workdir."""
  video_dir = os.path.join(workdir, "videos")
  if not os.path.exists(video_dir):
    os.makedirs(video_dir)
  rewards = {
      "blocktoblock":
          block2block.BlockToBlockReward,
      # "blocktoabsolutelocation":
      #     block2absolutelocation.BlockToAbsoluteLocationReward,
      # "blocktoblockrelativelocation":
      #     block2block_relative_location.BlockToBlockRelativeLocationReward,
      # "blocktorelativelocation":
      #     block2relativelocation.BlockToRelativeLocationReward,
      # "separate":
      #     separate_blocks.SeparateBlocksReward,
  }

  num_evals_per_reward = 10 #50
  max_episode_steps = 80 #200

  policy = None
  model,network_state = get_ckpt_model(ckpt)
  # keys = network_state.keys()
  # print("===========================================")
  # print(keys)
  # print("===========================================")

  results = collections.defaultdict(lambda: 0)
  for reward_name, reward_factory in rewards.items():
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        seed=0)
    env = gym_wrapper.GymWrapper(env)
    env = env_wrappers.UseTokenWrapper(env)
    env = env_wrappers.CentralCropImageWrapper(
        env,
        target_width=config["data_target_width"],
        target_height=config["data_target_height"],
        random_crop_factor=config["random_crop_factor"],)
    env = tfa_wrappers.HistoryWrapper(
        env, history_length=config["sequence_length"], tile_first_step_obs=True)

    if policy is None:
      policy = jax_policy.BCJaxPyPolicyRT1(
          env.time_step_spec(),
          env.action_spec(),
          model=model,
          network_state=network_state,
          rng=jax.random.PRNGKey(0))

    for ep_num in range(num_evals_per_reward):
      # Reset env. Choose new init if oracle cannot find valid motion plan.
      # Get an oracle. We use this at the moment to decide whether an
      # environment initialization is valid. If oracle can motion plan,
      # init is valid.

      policy.network_state['context_image_tokens'] = torch.zeros_like(policy.network_state['context_image_tokens'])
      policy.network_state['action_tokens'] = torch.zeros_like(policy.network_state['action_tokens'])
      policy.network_state['seq_idx'] = torch.zeros_like(policy.network_state['seq_idx'])

      oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
          env, use_ee_planner=True)
      plan_success = False
      while not plan_success:
        ts = env.reset()
        raw_state = env.compute_state()
        plan_success = oracle_policy.get_plan(raw_state)
        if not plan_success:
          logging.info(
              "Resetting environment because the "
              "initialization was invalid (could not find motion plan).")

      frames = [env.render()]

      episode_steps = 0
      while not ts.is_last():
        policy_step = policy.action(ts, ())
        ts = env.step(policy_step.action)
        frames.append(env.render())
        episode_steps += 1

        if episode_steps > max_episode_steps:
          break

      success_str = ""
      if env.succeeded:
        results[reward_name] += 1
        logging.info("Episode %d: success.", ep_num)
        success_str = "success"
      else:
        logging.info("Episode %d: failure.", ep_num)
        success_str = "failure"

      # Write out video of rollout.
      video_path = os.path.join(workdir, "videos/",
                                f"{reward_name}_{ep_num}_{success_str}.mp4")

      imageio.mimsave(video_path, frames, fps=10)

    print(results)


def main(argv):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model_config = {
      "data_target_width": 456,
      "data_target_height": 256,
      "random_crop_factor": 0.95,
      "sequence_length": 6,
  }

  evaluate_checkpoint(
      workdir=_WORKDIR.value,
      config=model_config,
      ckpt='/gemini/data-2/test/pytorch-RT1/exp/ckpt/exp_rt1/epoch=54-eval_loss=0.022458-train_loss_epoch=0.000278.ckpt',
  )


if __name__ == "__main__":
  
  app.run(main)




