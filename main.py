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

from model import M5  #model.py에 있는 M5 network를 import 해옴


class SubsetSC(SPEECHCOMMANDS): # speechcommand dataset이 없으면 다운로드 받고 학습 시키는게 train인지 test인지 판별함
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


def train(model, epoch, log_interval):
    losses = []
    pbar_update = 1 / (len(train_loader) + len(test_loader))

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # data = data.to(device)
        # target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def eval(model, epoch):
    pbar_update = 1 / (len(train_loader) + len(test_loader))
    model.eval()
    correct = 0
    for data, target in test_loader:
        # data = data.to(device)
        # target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)
    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


def save(model):
    torch.save({'model': model.state_dict()}, './checkpoints/last.pth')


def predict(tensor):
    # Use the model to predict the label of the waveform
    # tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


if __name__ == '__main__':
    global new_sample_rate

    # --------------------GPU 또는 CPU 사용-------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ------------------------------------------------------------

    # -------------------speechcommand데이터가 없으면 다운받고 train 과 test 데이터를 불러옴---------
    train_set = SubsetSC("training")  # number of train_data: 102859, number of classes: 35
    test_set = SubsetSC("testing")  # number of test_data: 11005, number of classes: 35
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
    # ---------------------------------------------------------------------

    # --------------------- sample_rate를 16KHz에서 8KHz로 변경 ----------------------------------
    new_sample_rate = 8000  # 8KHz로 사용
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)
    plt.plot(transformed.t().numpy())
    # -----------------------------------------------------------------------------------------

    # -------------------
    batch_size = 256
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    # ------------------------------------------------------

    # ------------------data_load-------------------------
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # -----------------------------------------------------

    # ---------------------- model ----------------------------
    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    # model.to(device)
    print(model)
    # -------------------------------------------------------------

    # ----------------optimizer, scheduler setup--------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # ---------------------------------------------------------------------------

    # -----------------training-------------------------
    log_interval = 20
    n_epoch = 5
    transform = transform.to(device)
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(model, epoch, log_interval)
            eval(model, epoch)
            scheduler.step()
            save(model)
    # ----------------------------------------------------

    # ----------------predict----------------------------------------

    for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
        output = predict(waveform)
        if output != utterance:
            print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
            break

        else:
            print("All examples in this dataset were correctly classified!")
            print("In this case, let's just look at the last data point")
            print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

    # ----------------------------------------------------------------

    #--------------------homework_predict-------------------------
    waveform, sample_rate, utterance, *_ = train_set[-1]
    print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")
    #-----------------------------------------------------------------
