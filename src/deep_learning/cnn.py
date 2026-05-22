#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
#%%
audio, sample_rate = librosa.load("data/silvera.wav")
print(audio.shape)
print(sample_rate)
#%%
#waveform->freq representation
mel_spec = librosa.feature.melspectrogram(
  y=audio,
  sr=sample_rate
)
print(mel_spec.shape)
#%%
#db scale
mel_spec_db = librosa.power_to_db(
  mel_spec,
  ref=np.max
)
#%%
plt.figure(figsize=(10,4))
librosa.display.specshow(
  mel_spec_db,
  sr=sample_rate,
  x_axis ='time',
  y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.show()
