import librosa

file_path = "data/bleed.wav"

audio, sample_rate = librosa.load(file_path)

mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

print("Audio loaded successfully!")
print("Sample Rate:", sample_rate)
print("Audio Length (samples):", len(audio))
print("MFCC shape:", mfccs.shape)