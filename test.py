import torch
import os
import torchaudio

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

def load_weights(option, model, optimizer, scheduler, device):
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

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

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
def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


if __name__ == '__main__':
    test_set = SubsetSC('testing')
    train_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    waveform, sample_rate, label, speaker_id, utterance_number = test_set[0]

    labels = sorted(list(set(datapoint[2] for datapoint in test_set)))
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
