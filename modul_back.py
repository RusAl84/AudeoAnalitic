import librosa
import librosa.display
import pandas as pd
import numpy as np
import os

def ext_mfcc(filename):
    audio, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    return mfcc_processed

if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\Asus\PycharmProjects\pythonProject4\jarchive\urbanSound8K.csv")
    features = []
    labels = []
    count = data.shape[0]
    for id, row in data.iterrows():
        file_name = os.path.join(os.path.abspath('jarchive/'), 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))
        features.append(ext_mfcc(file_name))
        labels.append(row['class'])
        print(f"{id} of {count}")

    np.save('features.npy', np.array(features))
    np.save('labels.npy', np.array(labels))
