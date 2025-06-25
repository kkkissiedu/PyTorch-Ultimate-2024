#%% package import
import torchaudio
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns
import matplotlib.pyplot as plt
#%% Check if audio backend is installed
torchaudio.info
# %% Wave Import
wav_file = 'data/set_a/extrahls__201101070953.wav'
data_waveform, sr = torchaudio.load(wav_file)
# %%
data_waveform.size()
# %% Plot Waveform
plot_waveform(data_waveform, sample_rate = sr)
# %% Calculate Spectogram
spectogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectogram.size()
# %% Plot Spectogram
plot_specgram(waveform = data_waveform, sample_rate = sr)
# %%
