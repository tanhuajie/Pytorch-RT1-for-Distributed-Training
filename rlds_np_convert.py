import rlds
import tensorflow_datasets as tfds
from tqdm import tqdm
import os
import numpy as np
import tensorflow_hub as hub
import copy

def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 

def create_episode(path,raw_episode,embed=False,embed_model=None,path_embed=None):
    episode, episode_embed = [], []
    for step in raw_episode[rlds.STEPS]:
        observation = step[rlds.OBSERVATION]
        observation_keys = observation.keys()
        step_keys = list(step.keys())
        step_keys.remove(rlds.OBSERVATION)
        step_dict = {}
        for k in step_keys:
            step_dict[k] = step[k].numpy()
        for k in observation_keys:
            step_dict[k] = observation[k].numpy()
        episode.append(step_dict)
        if embed:
            step_dict_embed = copy.deepcopy(step_dict)
            step_dict_embed["instruction"] = embed_model([decode_inst(step_dict_embed["instruction"])])[0].numpy()
            episode_embed.append(step_dict_embed)
    # save numpy
    np.save(path, episode)
    if embed:
        np.save(path_embed, episode_embed)

NUM_TRAIN = 7800
NUM_VAL = 100
NUM_TEST = 100

dataset_episode_num = 8000
builder = tfds.builder_from_directory(builder_dir="/gemini/data-2/data/language_table_b2b_sim")

ds = builder.as_dataset(
    split=f'train[{0}:{dataset_episode_num}]',
    #decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
    shuffle_files=False
)

embed_model = hub.load("/gemini/data-2/code/universal_sentence_encoder")

print("Generating train examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/train', exist_ok=True)
cnt = 0
for element in tqdm(ds.take(NUM_TRAIN)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/train/episode_{cnt}.npy', element,
                   embed=True, embed_model=embed_model, path_embed=f'/gemini/data-2/data/pytorch_b2b_sim_np/train/episode_{cnt}.npy')
    cnt = cnt + 1

print("Generating val examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/val', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN).take(NUM_VAL)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/val/episode_{cnt}.npy', element,
                   embed=True, embed_model=embed_model, path_embed=f'/gemini/data-2/data/pytorch_b2b_sim_np/val/episode_{cnt}.npy')
    cnt = cnt + 1

print("Generating test examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/test', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN + NUM_VAL).take(NUM_TEST)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/test/episode_{cnt}.npy', element,
                   embed=True, embed_model=embed_model, path_embed=f'/gemini/data-2/data/pytorch_b2b_sim_np/test/episode_{cnt}.npy')
    cnt = cnt + 1


