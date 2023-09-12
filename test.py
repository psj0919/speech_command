import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd
from model import M5
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os

from model import M5  #model.py에 있는 M5 network를 import 해옴
# 과제를 수행하기 위해 생성한 test코드
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def load_weights(option, model, optimizer, scheduler, device): # 저장된 가중치를 불러와 predict할 때 사용함
    if option.checkpoints_file is not None:
        print("Loading saved weights {}".format(option.checkpoints_file))
        file_path = os.path.join(option.checkpoints_save_path, option.checkpoints_file)
        if os.path.exists(file_path):
            weights = torch.load(file_path, map_location=device)
            model.load_state_dict(weights['model'])

        else:
            ValueError("There is no weight file!!")

    return model, optimizer, scheduler

def predict(tensor):
    # Use the model to predict the label of the waveform
    # tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def pad_sequence_for_pred(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def collate_fn_for_pred(batch):
    tensors = []
    for waveform in batch:
        tensors += [waveform]

    tensors = pad_sequence_for_pred(tensors)

    return tensors

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word)) # tensor형태로 labels.index(word)로 반환

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform] # 각 waveform에 대한 정보를 tensors라는 list에 저장
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors) # waveform에 대한 정보가 담김
    targets = torch.stack(targets) # label정보가 담김

    return tensors, targets


if __name__ == '__main__':
    # ----------training, test_data_load---------------------------------------------------
    train_set = SubsetSC("training")  # number of train_data: 102859, number of classes: 35
    test_set = SubsetSC("testing")  # number of test_data: 11005, number of classes: 35
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    new_sample_rate = 8000  # 8KHz로 사용
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate) #기존 sample_rate 16KHz에서 8KHz로 변경하기 위한 코드
    transformed = transform(waveform) #waveform의 sample_rate를 8KHz로 변경


    model = M5(n_input=transformed.shape[0], n_output=len(labels))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )  # train_set에 대한 데이터를 무작위로 load 해옴

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    ) # test_set에 대한 데이터를 무작위로 load 해옴
    # -------------------------------------------------------------------------------------------



    # ---------------------------------for predict---------------------------------------
    batch_size = 1

    pre_loader = torch.utils.data.DataLoader(
        './speech_test/_audio (1).wav',
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_for_pred,
        num_workers=0,
        pin_memory=False,
    )
    for id, data in enumerate(pre_loader):
        print(data)

    # ------------------------------------------------------------------------------------



