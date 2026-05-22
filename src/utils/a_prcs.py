#%%
import kagglehub
import os
#%%
def get_dataset_path():
  DATASET_PATH = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
  DATASET_PATH = os.path.join(DATASET_PATH, "Data", "genres_original")
  return DATASET_PATH