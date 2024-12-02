# Pytorch-RT1-for-Distributed-Training
A Repo for Pytorch-RT1 Distributed Training, and evaluating on Language-Table Simulator

## Install
```bash
conda create -n rtx --file requirements.txt
conda activate rtx
```

## Clone repo.
```bash
# clone git repo.
git clone https://github.com/tanhuajie/Pytorch-RT1-for-Distributed-Training.git
cd Pytorch-RT1-for-Distributed-Training
```

## Download Dataset

**Option1:** Download from [GoogleCloud](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_sim)

**Option2:** Download from [BaiDuYun](https://pan.baidu.com/s/1zpVy-IX48L-YOxkXPUDZCg?pwd=hsbw)

## Download Sentence Encoder

Download from [universal-sentence-encoder](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder)

```python
# Modify the path to universal_sentence_encoder in language_table/common/rt1_tokenizer.py

def tokenize_text(text):
  """Tokenizes the input text given a tokenizer."""
  embed = hub.load("/path/to/universal_sentence_encoder")
  tokens = embed([text])
  return tokens

```

## Convert Dataset
```bash
# convert dateset from RLDS to numpy, and split train/test/val
# Note: replace the paths with your own.
python rlds_np_convert.py

# evaluate your dataset to check whether it's ready for training or testing 
# Note: replace the paths with your own.
python load_np_dataset.py
```

## Distributed Train
```bash
# Note: replace the paths with your own.
python distribute_train.py --gpus 0,1,2,3 --lr 5e-4 --exp_name exp_rt1 --dataset_dir /path/to/your/dataset --log_dir /path/to/your/exp/logs --ckpt_dir /path/to/your/exp/ckpts
```

## Evaluation on Language-Table Simulator
```bash
# Note: replace the paths with your own.
python -m language_table.eval.main_rt1
```

