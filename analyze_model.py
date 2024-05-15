import librosa
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import models
# from tensorflow.keras.models import model
# from tensorflow.python import model

import matplotlib.animation as animation

labels = ['Кондиционер', 'Клаксон авто', 'Играющие дети', 'Лай собаки', 'Сверление', 'Двигатель', 'Выстрел',
          'Отбойный молоток', 'Сирена', 'Уличная музыка']

def predict(filename):
    audio, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.reshape(mfcc, (1, 40))
    model = models.load_model('best_model_Dense.h5')

    pred = model.predict(mfcc)
    #print(np.argmax(pred[0]))
    return "Звук, который был распознан - " + labels[np.argmax(pred[0])]

def gen_gif(filename):
    audio, sr = librosa.load(filename)

    model = models.load_model('best_model_Dense.h5')

    if len(audio)/sr > 1:
        length = round(len(audio) / sr)
        sections = np.linspace(0, length, num=length * 2 + 1)
        mfccs_ = []

        for i in sections[:-1]:
            mfcc = librosa.feature.mfcc(y=audio[round(i * sr):round((i + 1) * sr)],
                                        sr=sr,
                                        n_mfcc=40)
            mfcc = np.mean(mfcc.T, axis=0)
            mfcc = np.reshape(mfcc, (1, 40))
            # mfcc = np.expand_dims(mfcc, axis=2)
            mfccs_.append(mfcc)

        predicts = []
        for i in mfccs_:
            predicts.append(model.predict(i))

    else:
        mfcc = librosa.feature.mfcc(y=audio,
                                    sr=sr,
                                    n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = np.reshape(mfcc, (1, 40))
        # mfcc = np.expand_dims(mfcc, axis=2)
        predicts = []
        predicts.append(model.predict(mfcc))

    ani = animation.FuncAnimation(fig, animate,
                                  frames=predicts,
                                  interval=500,
                                  repeat=False)
    ani.save('graph.gif',
             writer='imagemagick')

    return 'graph.gif'

fig, ax = plt.subplots(figsize=(7, 4))

def animate(i):
    ax.clear()
    pr = np.argmax(i[0])
    bar = ax.barh(labels, i[0])
    plt.xticks(np.arange(0, 1.2, step=0.2))
    #ax.set_yticklabels(labels, fontsize=8, rotation=45)
    lb = plt.yticks(range(10), labels, rotation=45)
    plt.title(labels[pr])
    plt.tight_layout()
    return bar
