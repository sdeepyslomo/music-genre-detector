#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.a_prcs import get_dataset_path
#%%
#function to convert .wav -> melspec
def extract_mel_spec(file_path):
  audio, sample_rate = librosa.load(file_path)
   #%%
  #waveform->freq representation
  mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate
  )
  #%%
  #db scale
  mel_spec_db = librosa.power_to_db(
    mel_spec,
    ref=np.max
  )
  return mel_spec_db,sample_rate
#%%
spectrograms=[]
labels=[]
#%%
DATASET_PATH = get_dataset_path()
for genre in os.listdir(DATASET_PATH):
  genre_path = os.path.join(DATASET_PATH, genre)
  if not os.path.isdir(genre_path):
    continue
  for file in os.listdir(genre_path):
    file_path = os.path.join(genre_path,file)
    try:
      spec,sample_rate = extract_mel_spec(file_path)
      spec=spec[:,:1500]
      spectrograms.append(spec)
      labels.append(genre)
    except Exception as e:
      print(f"Error processing {file_path}: {e}")
#%%
spec,sample_rate= extract_mel_spec("data/silvera.wav")
print(spec.shape)
#%%
plt.figure(figsize=(10,4))
librosa.display.specshow(
  spec,
  sr=sample_rate,
  x_axis ='time',
  y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.show()



