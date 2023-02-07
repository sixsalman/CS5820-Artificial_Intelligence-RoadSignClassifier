"""
This script reads and augments images of 30, 50, 60, 70, 80, 100 and 120 Km/h German speed limit signs from
GTRSBSubsetSplitted directory before using RandomSearch to train a variety of CNN models over them. Upto 108
combinations of hyperparameters are trained and training is stopped, through a callback, if accuracy > 95% and
loss < 0.1 with less than 1% difference in validation values is reached. If previous results of RandomSearch already
exist in the directory RoadSignModels, training is skipped. After a model is ready, user is asked for path to a picture
of a speed limit sign for classification.
"""

from silence_tensorflow import silence_tensorflow
import tensorflow.keras.preprocessing.image as image
from keras_tuner import RandomSearch
import tensorflow as tf
import numpy as np


def main():
    silence_tensorflow()

    training_data = image.ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.1,
                                             height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                             fill_mode='nearest') \
        .flow_from_directory("GTRSBSubsetSplitted/training/", target_size=(62, 62), class_mode='categorical')

    validation_data = image.ImageDataGenerator(rescale=1.0 / 255) \
        .flow_from_directory("GTRSBSubsetSplitted/validation/", target_size=(62, 62), class_mode='categorical')

    tuner = RandomSearch(build_model, objective='val_loss', project_name='RoadSignModels', max_trials=108)
    tuner.search(training_data, epochs=25, validation_data=validation_data, callbacks=[CbEarlyEndCheck()])

    model = tuner.get_best_models(2)[1]

    output = model.predict(np.expand_dims(image.img_to_array(image.load_img(input("\nEnter picture path: "),
                                                                            target_size=(62, 62))), axis=0) / 255.0)[0]
    print(f"\nClasses: {list(training_data.class_indices.keys())}\nConfidences (out of 1): {output}\n\n"
          f"Class with highest confidence: {list(training_data.class_indices.keys())[np.argmax(output)]}\n"
          f"Confidence: {(output[np.argmax(output)] * 100):.2f}%")


def build_model(hp):
    """
    Builds structures with 2 or 3 convolutional layers with combinations of 16, 32 or 64 filters in each followed by a
    dense layer with 64, 128 or 256 units.
    :param hp: used for hyperparameter variation
    :return: the built model
    """
    model_structure = tf.keras.models.Sequential()

    model_structure.add(tf.keras.layers.Conv2D(hp.Choice('Conv2D_1_filters', values=[16, 32, 64]), (3, 3),
                                               activation='relu', input_shape=(62, 62, 3)))
    model_structure.add(tf.keras.layers.MaxPooling2D(2, 2))

    model_structure.add(tf.keras.layers.Conv2D(hp.Choice('Conv2D_2_filters', values=[16, 32, 64]), (3, 3),
                                               activation='relu'))
    model_structure.add(tf.keras.layers.MaxPooling2D(2, 2))

    conv3_filters = hp.Choice('Conv2D_3_filters', values=[0, 16, 32, 64])
    if conv3_filters != 0:
        model_structure.add(tf.keras.layers.Conv2D(conv3_filters, (3, 3), activation='relu'))
        model_structure.add(tf.keras.layers.MaxPooling2D(2, 2))

    model_structure.add(tf.keras.layers.Flatten())
    model_structure.add(tf.keras.layers.Dense(hp.Choice('dense_units', values=[64, 128, 256]), activation='relu'))
    model_structure.add(tf.keras.layers.Dense(7, activation='softmax'))

    model_structure.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(),
                            metrics=['accuracy'])

    return model_structure


class CbEarlyEndCheck(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') > 0.95 and abs(logs.get('accuracy') - logs.get('val_accuracy')) < 0.01 and
                logs.get('loss') < 0.1 and abs(logs.get('loss') - logs.get('val_loss')) < 0.01):
            print("\nReached accuracy > 95% and loss < 0.1 with less than 1% difference in validation values. "
                  "Stopping training for this hyperparameter combination.\n")

            self.model.stop_training = True


main()
