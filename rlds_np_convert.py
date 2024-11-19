import rlds
import tensorflow_datasets as tfds
from tqdm import tqdm
import os
import numpy as np

def create_episode(path,raw_episode):
    episode = []
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
    np.save(path, episode)

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

print("Generating train examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/train', exist_ok=True)
cnt = 0
for element in tqdm(ds.take(NUM_TRAIN)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/train/episode_{cnt}.npy',element)
    cnt = cnt + 1

print("Generating val examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/val', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN).take(NUM_VAL)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/val/episode_{cnt}.npy', element)
    cnt = cnt + 1

print("Generating test examples...")
os.makedirs('/gemini/data-2/data/inst_pytorch_b2b_sim_np/test', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN + NUM_VAL).take(NUM_TEST)):
    create_episode(f'/gemini/data-2/data/inst_pytorch_b2b_sim_np/test/episode_{cnt}.npy', element)
    cnt = cnt + 1


