import librosa
import os
import torchaudio
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt

if __name__=='__main__':
    wav = './speech_test/_audio (1).wav'
    (file_dir, file_id) = os.path.split(wav)
    print("file_dir:", file_dir)
    print("file_id:", file_id)

    wavform, sample_rate = torchaudio.load(wav)
    plt.plot(wavform.t().numpy())
    transform = T.Resample(orig_freq=43200, new_freq=16000)
    wavform = transform(wavform)

    print(wavform.shape)
    wavform2 = wavform[6400:14400]
    # librosa.output.write_wav('./sppech_test/cut_file.wav', wavform2, sample_rate)
