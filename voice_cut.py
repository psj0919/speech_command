import librosa
import os
import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt
import torchaudio.transforms as T

if __name__=='__main__':
    wav = './speech_test/_audio (1).wav'
    (file_dir, file_id) = os.path.split(wav)
    #print("file_dir:", file_dir)
    #print("file_id:", file_id)

    # y, sr = librosa.load(wav, sr=16000)
    # time = np.linspace(0, len(y) / sr, len(y))  # time axis
    #
    # fig, ax1 = plt.subplots()  # plot
    #
    # # cut half and save
    # half = len(y) / 2
    # y2 = y[6400:14400]
    # time2 = np.linspace(0, len(y2) / sr, len(y2))
    # fig2, ax2 = plt.subplots()
    # ax2.plot(time2, y2, color='b', label='speech waveform')
    # ax1.set_ylabel("Amplitude")  # y 축
    # ax1.set_xlabel("Time [s]")  # x 축
    # plt.title('cut ' + file_id)
    # plt.savefig('cut_half ' + file_id + '.png')
    # plt.show()
    # #librosa.output.write_wav('./sppech_test/cut_file.wav', y2, sr)


    wavform, sample_rate = torchaudio.load(wav)
    transform = T.Resample(orig_freq=43200, new_freq=16000)
    wavform = transform(wavform)

    print(wavform.shape)
    reshaped_waveform = wavform.view(1, -1)
    print(reshaped_waveform.shape)