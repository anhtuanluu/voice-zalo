import librosa
import numpy as np
from python_speech_features import fbank
import configure as c
from DB_wav_reader import read_DB_structure
import pickle

def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

# audiopath = c.TRAIN_WAV_DIR
audiopath = '/home/tuan/Documents/Train-Test-Data/public-test'
db = read_DB_structure(audiopath)

feat_and_label={}

print(db["filename"])
for filename, label in zip(db["filename"], db["speaker_id"]):
    print(filename)
    print(label)
    audio, sr = librosa.load(filename, sr=c.SAMPLE_RATE, mono=True)
    filter_banks, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=40, winlen=0.025)
    filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    feature = normalize_frames(filter_banks, Scale=False)
    feat_and_label['feat'] = feature
    feat_and_label['label'] = label
    save_file = filename[:-4] + ".p"
    with open(save_file, "wb") as f:
        pickle.dump(feat_and_label, f)
