import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os

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

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def resample_rate(rate, n_rate):

    transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=n_rate)
    transformed = transform(waveform)
    return transformed

if __name__=='__main__':
    # --------------------GPU 또는 CPU 사용-------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ------------------------------------------------------------

    # -------------------speechcommand데이터가 없으면 다운받고 train 과 test 데이터를 불러옴---------
    train_set = SubsetSC("training") # number of train_data: 102859, number of classes: 35
    test_set = SubsetSC("testing") # number of test_data: 11005, number of classes: 35
    # -------------------------------------------------------------------------------------

    # ---------------------------------dataset------------------------------------
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))
    plt.plot(waveform.t().numpy())
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    # ----------------------------------------------------------------------------

    # ------------------- 음성데이터 들어보기 위한 예제 코드 -----------------------
    waveform_first, *_ = train_set[0]
    ipd.Audio(waveform_first.numpy(), rate=sample_rate)

    waveform_second, *_ = train_set[1]
    ipd.Audio(waveform_second.numpy(), rate=sample_rate)

    waveform_last, *_ = train_set[-1]
    ipd.Audio(waveform_last.numpy(), rate=sample_rate)
    # --------------------------------------------------------------------


    # --------------------- sample_rate를 16KHz에서 8KHz로 변경 ----------------------------------
    transformed = resample_rate(16000,8000)
    plt.plot(transformed.t().numpy())
    ipd.Audio(transformed.numpy(), rate=8000)
    # --------------------------------------------------------------------

    # --------------------
    word_start = "yes"
    index = label_to_index(word_start)
    word_recovered = index_to_label(index)

    print(word_start, "-->", index, "-->", word_recovered)
