import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from argparse import ArgumentParser
from pytorch_robotics_transformer.transformer_network import TransformerNetwork
from gym import spaces
import numpy as np
from collections import OrderedDict
from pytorch_robotics_transformer.tokenizers.utils import batched_space_sampler
from pytorch_robotics_transformer.tokenizers.utils import np_to_tensor
from load_np_dataset import DecodeAndRandomResizedCrop, EmbodiedIntelligenceDataset
from load_np_dataset import collate_fn


class RT1_Lightning(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.HEIGHT = self.args.height
        self.WIDTH = self.args.width
        self.TIME_SEQUENCE_LENGTH = self.args.seq_len

        self.state_space = spaces.Dict(
            {
                'image': spaces.Box(low=0.0, high=1.0, shape=(3, self.HEIGHT, self.WIDTH), dtype=np.float32),
                'natural_language_embedding': spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32)
            }
        )

        self.action_space = spaces.Dict(
            OrderedDict([
                ('terminate_episode', spaces.Discrete(2)), 
                ('action', spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)),
            ])
        )

        self.model = TransformerNetwork(
            input_tensor_space = self.state_space,
            output_tensor_space = self.action_space,
            vocab_size = 256,
            token_embedding_size = 512,
            num_layers = 8,
            layer_size = 128,
            num_heads = 8,
            feed_forward_size = 512,
            dropout_rate = 0.1,
            time_sequence_length = self.TIME_SEQUENCE_LENGTH,
            crop_size = 236,
            use_token_learner = True
        )

        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        observations = batch['train_observation']
        action_labels = batch['action_label']
        self.model.set_actions(action_labels)
        network_states = batched_space_sampler(self.model._state_space, batch_size=action_labels['action'].shape[0])
        network_states = np_to_tensor(network_states, 'cuda')
        self.model(observations, network_states)
        # pred loss
        loss = self.loss_fn(self.model.get_actor_loss())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True) 

        # self.epoch_loss = self.epoch_loss + loss

        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        observations = batch['train_observation']
        action_labels = batch['action_label']
        self.model.set_actions(action_labels)
        network_states = batched_space_sampler(self.model._state_space, batch_size=action_labels['action'].shape[0])
        network_states = np_to_tensor(network_states, 'cuda')
        self.model(observations, network_states)
        # pred loss
        loss = self.loss_fn(self.model.get_actor_loss())
        self.log("test_loss", loss, prog_bar=True) 

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        observations = batch['train_observation']
        action_labels = batch['action_label']
        self.model.set_actions(action_labels)
        network_states = batched_space_sampler(self.model._state_space, batch_size=action_labels['action'].shape[0])
        network_states = np_to_tensor(network_states, 'cuda')
        self.model(observations, network_states)
        # pred loss
        loss = self.loss_fn(self.model.get_actor_loss())
        self.log("eval_loss", loss, prog_bar=True) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.milestones, gamma=0.1, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "interval": "epoch",
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                },
            }
    
    def loss_fn(self, x):
        '''
            define loss function here
        '''
        loss = torch.mean(x)

        return loss

    # def on_train_epoch_start(self):
    #     self.epoch_loss = 0

    # def on_train_epoch_end(self):
    #     self.epoch_loss = 0

    def debug(self, training = True):

        BATCH_SIZE = 2

        if training:
            image_shape = [BATCH_SIZE, self.TIME_SEQUENCE_LENGTH, 3, self.HEIGHT, self.WIDTH]
            emb_shape = [BATCH_SIZE, self.TIME_SEQUENCE_LENGTH, 512]
            action_shape = [BATCH_SIZE, self.TIME_SEQUENCE_LENGTH, 2]
            terminata_shape = [BATCH_SIZE, self.TIME_SEQUENCE_LENGTH]
        else:
            # inference currently only support batch size of 1
            image_shape = [1, 3, self.HEIGHT, self.WIDTH]
            emb_shape = [1, 512]
            action_shape = [1, 2]
            terminata_shape = [1]

        batch = {
            'observation': {
                'image': torch.full(image_shape, 0.5, dtype=torch.float32).to('cuda'),
                'natural_language_embedding': torch.full(emb_shape, 1.0, dtype=torch.float32).to('cuda')
            },
            'action_label':{
                'terminate_episode': torch.full(terminata_shape, 1).to('cuda'), 
                'action': torch.full(action_shape, 0.0, dtype=torch.float32).to('cuda')
            }
        }

        observations = batch['observation']
        action_labels = batch['action_label']

        self.model.to('cuda')

        if training:
            print(f"action: {action_labels['action'].shape}")
            print(f"is_terminal: {action_labels['terminate_episode'].shape}")
            self.model.set_actions(action_labels)

        print(f"natural_language_embedding: {observations['natural_language_embedding'].shape}")
        print(f"image: {observations['image'].shape}")

        network_states = batched_space_sampler(self.model._state_space, batch_size=action_labels['action'].shape[0])
        network_states = np_to_tensor(network_states, 'cuda')

        print(f"network_state idx:{network_states['seq_idx']}")
        print(f"network_state action_tokens:{network_states['action_tokens'].shape}")
        print(f"network_state context_image_tokens:{network_states['context_image_tokens'].shape}")

        pred_action, _ = self.model(observations, network_states)

        if training:
            action_loss = self.model.get_actor_loss()
            mean_loss = self.loss_fn(action_loss)
            return pred_action, action_loss, mean_loss
        else:
            return pred_action

        

def print_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print_keys(value, prefix + key + ".")
    else:
        print(prefix[:-1])


def train(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    transform = DecodeAndRandomResizedCrop(random_crop_factor=args.random_crop_factor, resize_size=(args.width, args.height))

    print("Loading Train-Dataset ...")
    train_dataset = EmbodiedIntelligenceDataset(os.path.join(args.dataset_dir,'train'), list(range(args.train_episode)), args.seq_len, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    print("Loading Test-Dataset ...")
    test_dataset = EmbodiedIntelligenceDataset(os.path.join(args.dataset_dir,'test'), list(range(args.test_episode)), args.seq_len, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    print("Loading Eval-Dataset ...")
    eval_dataset = EmbodiedIntelligenceDataset(os.path.join(args.dataset_dir,'val'), list(range(args.eval_episode)), args.seq_len, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)


    # model
    print("Buliding RT1 Model ...")
    model = RT1_Lightning(args)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
            dirpath = os.path.join(args.ckpt_dir, args.exp_name),
            save_top_k = -1,
            filename = '{epoch}-{eval_loss:.6f}-{train_loss_epoch:.6f}',
            save_last = True,
            every_n_epochs = args.ckpt_every_n_epochs
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor]

    # loggers
    csv_logger = CSVLogger(save_dir = os.path.join(args.log_dir,'csv'), name = args.exp_name)
    tb_logger = TensorBoardLogger(save_dir = os.path.join(args.log_dir,'tb'), name = args.exp_name)

    loggers = [csv_logger, tb_logger]

    # train model
    trainer = Trainer(
        accelerator = args.device, 
        max_epochs = args.max_epochs, 
        log_every_n_steps = args.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks = callbacks, 
        logger = loggers
    )
    
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt") # resume
    # print("Start Initial Testing:")
    # trainer.test(model, test_loader)

    print("Start Training ...")
    trainer.fit(model, train_loader, eval_loader)
    print("Start Final Testing ...")
    trainer.test(model, test_loader)


def debug(args):

    agent = RT1_Lightning(args)

    print(f"=========Training Debug=========")  

    pred_action, action_loss, mean_loss = agent.debug(training=True)

    print(f"pred_action:{pred_action['action'].shape}")
    print(f"is_terminate:{pred_action['terminate_episode'].shape}")
    print(f"action_loss:{action_loss.shape}, mean_loss:{mean_loss.shape}")

    print(f"=========Inference Debug=========")
    pred_action = agent.debug(training=False)

    print(f"pred_action:{pred_action['action'].shape}")
    print(f"is_terminate:{pred_action['terminate_episode'].shape}")


if __name__ == "__main__":
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

    # train(args)
    debug(args)