import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns

# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model

from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD


def network_train():
    X = np.load('features.npy')
    y = np.load('labels.npy')

    lb = LabelEncoder()
    yy = to_categorical(lb.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, shuffle=True, random_state=331)

    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(40,)))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    # plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_callback = EarlyStopping(monitor='val_accuracy',
                                            patience=10)
    checkpoint_filepath = 'best_model_Dense.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    print(X_test.shape)
    history = model.fit(X_train,
                        y_train,
                        batch_size=32,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        callbacks=[model_checkpoint_callback,
                                   early_stopping_callback])

    plt.title('Losses')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.show()

    model = models.load_model('best_model_Dense.h5')
    _, test_acc = model.evaluate(X_test, y_test)
    print('after_load_acc: ', test_acc)

    preds = model.predict(X_test,
                          batch_size=32,
                          verbose=1)
    preds = preds.argmax(axis=1)
    preds = preds.astype(int).flatten()
    preds = (lb.inverse_transform((preds)))
    preds = pd.DataFrame({'predicted_values': preds})

    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'actual_values': actual})

    df = actual.join(preds)
    c = confusion_matrix(df.actual_values, df.predicted_values)
    print(accuracy_score(df.actual_values, df.predicted_values))

    df_cm = pd.DataFrame(c, index=lb.classes_, columns=lb.classes_)
    fig = plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap=plt.cm.Blues)

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    network_train()