import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class DecodeAndRandomResizedCrop():
    def __init__(self, random_crop_factor=None, resize_size=(456, 256)):
        self.random_crop_factor = random_crop_factor
        self.resize_size = resize_size

    def __call__(self, image):
        raw_width, raw_height = image.size

        if self.random_crop_factor is None:
            random_crop_factor = 1.0
            offset_width = 0
            offset_height = 0
            scaled_height = raw_height
            scaled_width = raw_width
        else:
            random_crop_factor = self.random_crop_factor
            scaled_height = raw_height * random_crop_factor
            scaled_width = raw_width * random_crop_factor

            offset_height = np.random.randint(0, raw_height - scaled_height + 1)
            offset_width = np.random.randint(0, raw_width - scaled_width + 1)

        crop_box = (offset_width, offset_height, offset_width + scaled_width, offset_height + scaled_height)
        # print(f"orig_image:{image.size}")
        image = image.crop(crop_box)
        # print(f"crop_image:{image.size}")
        image = image.resize(self.resize_size, Image.BILINEAR)
        # print(f"resize_image:{image.size}")
        image = np.array(image).astype(np.float32) / 255.0
        # print(f"tensor_image:{torch.tensor(image).shape}")
        # print(f"tensor_image:{torch.tensor(image).permute(2, 0, 1).shape}")
        return torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW format

class EmbodiedIntelligenceDataset(Dataset):
    def __init__(self, data_dir, ids, window_length, transform=None):
        self.data_dir = data_dir
        self.ids = ids
        self.window_length = window_length
        self.transform = transform
        self.samples = self._create_samples()

    def _pad_episode(self, episode):
        padding = self.window_length - 1
        first_item = episode[0]
        padding_item = first_item.copy()
        padding_item["is_first"] = False

        # Create padding steps by copying the first step
        padding_steps = [padding_item for _ in range(padding)]

        # Concatenate padding steps with the original steps
        episode = [first_item] + padding_steps + episode[1:].tolist()

        # print(len(episode))

        return episode

    def _create_samples(self):
        samples = []
        for episode_id in self.ids:
            file_path = os.path.join(self.data_dir, f'episode_{episode_id}.npy')
            episode = np.load(file_path, allow_pickle=True)
            episode = self._pad_episode(episode)
            num_steps = len(episode)
            for i in range(num_steps - self.window_length + 1):
                samples.append((episode_id, i))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_id, start_idx = self.samples[idx]
        file_path = os.path.join(self.data_dir, f'episode_{episode_id}.npy')
        episode = np.load(file_path, allow_pickle=True)
        episode = self._pad_episode(episode)
        
        observations = []
        actions = []
        is_terminals = []
        
        for i in range(start_idx, start_idx + self.window_length):
            step = episode[i]
            observation = step["rgb"]
            if self.transform:
                observation = self.transform(Image.fromarray(observation))
            else:
                observation = torch.tensor(observation).permute(2, 0, 1).float() / 255.0  # Convert to CxHxW format
            observations.append({
                "image": observation,
                "natural_language_embedding": torch.tensor(step["instruction"], dtype=torch.float32)
            })
            actions.append(torch.tensor(step["action"], dtype=torch.float32))
            if step["is_terminal"] == True:
                is_terminals.append(torch.tensor(1))
            else:
                is_terminals.append(torch.tensor(0))
        
        action_label = {
            "terminate_episode": torch.stack(is_terminals),
            "action": torch.stack(actions)
        }
        train_observation = {
            "image": torch.stack([obs["image"] for obs in observations]),
            "natural_language_embedding": torch.stack([obs["natural_language_embedding"] for obs in observations])
        }
        data = {"action_label": action_label, "train_observation": train_observation}
        
        return data
    
def debug(data_dir, episode_id, step_id):
    file_path = os.path.join(data_dir, f'episode_{episode_id}.npy')
    episode = np.load(file_path, allow_pickle=True)
    step = episode[step_id]
    # print(step.keys())
    print(f"action: {step['action']}")
    print(f"is_first: {step['is_first']}")
    print(f"is_terminal: {step['is_terminal']}")
    print(f"instruction: {step['instruction'].shape}")
    print(f"rgb: {step['rgb'].shape}")



def collate_fn(batch):
    action_labels = {"terminate_episode": [], "action": []}
    train_observations = {"image": [], "natural_language_embedding": []}

    for item in batch:
        action_labels["terminate_episode"].append(item["action_label"]["terminate_episode"])
        action_labels["action"].append(item["action_label"]["action"])
        train_observations["image"].append(item["train_observation"]["image"])
        train_observations["natural_language_embedding"].append(item["train_observation"]["natural_language_embedding"])

    action_labels["terminate_episode"] = torch.stack(action_labels["terminate_episode"])
    action_labels["action"] = torch.stack(action_labels["action"])
    train_observations["image"] = torch.stack(train_observations["image"])
    train_observations["natural_language_embedding"] = torch.stack(train_observations["natural_language_embedding"])

    return {"action_label": action_labels, "train_observation": train_observations}

if __name__ == "__main__":

    # debug('/gemini/data-2/data/pytorch_b2b_sim_np/test', 0, 0)

    data_dir = '/gemini/data-2/data/pytorch_b2b_sim_np/test'

    ids = list(range(50))

    window_length = 6  # Example window length, can be adjusted

    transform = DecodeAndRandomResizedCrop(random_crop_factor=0.95, resize_size=(456, 256))

    dataset = EmbodiedIntelligenceDataset(data_dir, ids, window_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Example usage:
    for batch in dataloader:
        data = batch
        action_label = data["action_label"]
        train_observation = data["train_observation"]
        
        print(train_observation["image"].shape)  # Expected shape: (batch_size, window_length, 3, resize_height, resize_width)
        print(train_observation["natural_language_embedding"].shape)  # Expected shape: (batch_size, window_length, ...)
        print(action_label["terminate_episode"].shape)  # Expected shape: (batch_size, window_length)
        print(action_label["action"].shape)  # Expected shape: (batch_size, window_length)

        pil_image = Image.fromarray((train_observation["image"][1,1,:,:,:].reshape([3, 256, 456]).permute(1, 2, 0) * 255).byte().numpy(), 'RGB')
        pil_image.save('./imgs.png')

        # print(train_observation["image"])  # Expected shape: (batch_size, window_length, 3, resize_height, resize_width)
        # print(train_observation["natural_language_embedding"])  # Expected shape: (batch_size, window_length, ...)
        # print(action_label["terminate_episode"])  # Expected shape: (batch_size, window_length)
        # print(action_label["action"])  # Expected shape: (batch_size, window_length)

        break