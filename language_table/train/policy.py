# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyPolicy for BC Jax."""

from flax.training import checkpoints
# import jax
# import jax.numpy as jnp
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
import tensorflow as tf
from pytorch_robotics_transformer.tokenizers.utils import np_to_tensor

EPS = np.finfo(np.float32).eps

import torch


class BCJaxPyPolicyRT1(py_policy.PyPolicy):
  """Runs inference with a BC policy."""

  def __init__(self, time_step_spec, action_spec, model, network_state, rng):
    super(BCJaxPyPolicyRT1, self).__init__(time_step_spec, action_spec)
    self.model = model
    self.rng = rng
    self.network_state = network_state
    # self._run_action_inference_jit = jax.jit(self._run_action_inference)
    self._run_action_inference_jit = self._run_action_inference


    self.action_std = 1.0
    self.action_mean = 0.0
    self.action_minimum = -0.03
    self.action_maximum =  0.03
    
    self.pre_pos = np.array([0.00,0.00])
    self.dst_pos = np.array([0.00,0.00])

    self.rt1_observation = {
      'image': None,
      'natural_language_embedding': None
    }

  def _run_action_inference(self, observation):

    # # Add a batch dim.
    # observation = jax.tree.map(lambda x: jnp.expand_dims(x, 0), observation)

    # print(observation['rgb_sequence'].shape)
    # print(observation['instruction_tokenized_use'].shape)

    self.rt1_observation['image'] = torch.tensor(observation['rgb_sequence'][-1,:,:,:].reshape([1,256,456,3])).permute([0,3,1,2]).to('cuda')
    self.rt1_observation['natural_language_embedding'] = torch.tensor(observation['instruction_tokenized_use'][-1,:].reshape([1,512])).to('cuda')
    
    #self.pre_pos = observation['effector_translation'][-1,:].reshape([2,])
    
    # self.rt1_observation['image'] = torch.tensor(observation['rgb_sequence'].reshape([1,6,256,456,3])).permute([0,1,4,2,3]).to('cuda')
    # self.rt1_observation['natural_language_embedding'] = torch.tensor(observation['instruction_tokenized_use'].reshape([1,6,512])).to('cuda')

    # print(self.rt1_observation)
    print("==========enter=========")
    print(f"BEFORE network_state idx:{self.network_state['seq_idx']}")

    # print(f"natural_language_embedding: {self.rt1_observation['natural_language_embedding'].shape}")
    # print(f"image: {self.rt1_observation['image'].shape}")
    # print(f"BEFORE network_state action_tokens:{self.network_state['action_tokens'].shape}")
    # print(f"BEFORE network_state context_image_tokens:{self.network_state['context_image_tokens'].shape}")
    with torch.no_grad():
      pred_action, self.network_state = self.model.model(self.rt1_observation, self.network_state)
      self.network_state['seq_idx'] = torch.tensor(self.network_state['seq_idx']).unsqueeze(0)
      torch.cuda.empty_cache()
    # print("==========exit========")

    pred_effector_target_translation = np.array(pred_action["action"].to("cpu"))
    pred_terminate_episode = np.array(pred_action["terminate_episode"].to("cpu"))
    
    # print(f"AFTER network_state:{self.network_state['seq_idx']}")
    # print(f"pred_action:{pred_effector_target_translation}, terminate:{pred_terminate_episode}")

    action = pred_effector_target_translation * np.maximum(self.action_std, EPS) + self.action_mean

    # Clip the action to spec.
    action = np.clip(action, self.action_minimum, self.action_maximum)
    
    # print(f"clip_action:{action}")

    return action

  def _action(self, time_step, policy_state=(), seed=0):
    observation = time_step.observation
    action = self._run_action_inference_jit(observation)[0]
    self.go_action = action
    #self.dst_pos = action.copy()
    #self.go_action = self.dst_pos - self.pre_pos
    #print(f"self.dst_pos:{self.dst_pos}")
    #print(f"self.pre_pos:{self.pre_pos}")
    print(f"self.go_action:{self.go_action}")
    #self.pre_pos = self.dst_pos.copy()
    return policy_step.PolicyStep(action=self.go_action)

# class BCJaxPyPolicy(py_policy.PyPolicy):
#   """Runs inference with a BC policy."""

#   def __init__(self, time_step_spec, action_spec, model, checkpoint_path,
#                rng, params=None, action_statistics=None):
#     super(BCJaxPyPolicy, self).__init__(time_step_spec, action_spec)
#     self.model = model
#     self.rng = rng

#     if params is not None and action_statistics is not None:
#       variables = {
#           "params": params,
#           "batch_stats": {}
#       }
#     else:
#       state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
#       variables = {
#           "params": state_dict["params"],
#           "batch_stats": state_dict["batch_stats"]
#       }

#     if action_statistics is not None:
#       self.action_mean = np.array(action_statistics["mean"])
#       self.action_std = np.array(action_statistics["std"])
#     else:
#       # We can load the observation and action statistics from the state dict.
#       self.action_mean = np.array(
#           state_dict["norm_info"]["action_statistics"]["mean"])
#       self.action_std = np.array(
#           state_dict["norm_info"]["action_statistics"]["std"])

#       self._rgb_mean = jnp.array(
#           state_dict["norm_info"]["observation_statistics"]["rgb"]["mean"])
#       self._rgb_std = jnp.array(
#           state_dict["norm_info"]["observation_statistics"]["rgb"]["std"])

#     self.variables = variables

#     self._run_action_inference_jit = jax.jit(self._run_action_inference)

#   def _run_action_inference(self, observation):
#     # Add a batch dim.
#     observation = jax.tree.map(lambda x: jnp.expand_dims(x, 0), observation)

#     normalized_action = self.model.apply(
#         self.variables, observation, train=False)
#     action = (
#         normalized_action * jnp.maximum(self.action_std, EPS) +
#         self.action_mean)

#     # Clip the action to spec.
#     action = jnp.clip(action, self.action_spec.minimum,
#                       self.action_spec.maximum)

#     return action

#   def _action(self, time_step, policy_state=(), seed=0):
#     observation = time_step.observation
#     action = self._run_action_inference_jit(observation)[0]
#     return policy_step.PolicyStep(action=action)
