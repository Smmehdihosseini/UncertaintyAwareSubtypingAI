import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np

class SaveCallback(Callback):

    def __init__(self, logger, save_path, save_epoch=10, save_weights_only=True):

        super(SaveCallback, self).__init__()
        self.logger = logger
        self.save_path = save_path
        self.save_weights_only = save_weights_only
        self.save_epoch = save_epoch

    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.save_epoch == 0:

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            filepath = os.path.join(self.save_path, f'model_epoch_{epoch + 1}.weights.h5')
            
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)

            self.logger.info(f'++++++++> [SaveCallback] Weights Regulary Saved at Epoch {epoch + 1} to {filepath}')

class MetricsLoggerCallback(Callback):

    def __init__(self, logger):

        super(MetricsLoggerCallback, self).__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        
        loss = logs.get('loss')
        accuracy = logs.get('accuracy')
        self.logger.info(f'--------> Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}')

class BestModelCallback(Callback):
    def __init__(self, logger, save_path, monitor='loss', save_weights_only=True):

        super(BestModelCallback, self).__init__()
        self.logger = logger
        self.save_path = save_path
        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.best_val = None

        if 'loss' in self.monitor:
            self.mode = 'min'
        elif 'accuracy' in self.monitor:
            self.mode = 'max'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def on_epoch_end(self, epoch, logs=None):

        val = logs.get(self.monitor)

        if val is None:
            self.logger.warning(f"--------> Monitor Value for {self.monitor} is Not Available!")
            return
        
        filepath = os.path.join(self.save_path, f'best_weights.weights.h5')

        if self.best_val is None:
            self.best_val = val
            self._save_model(epoch, filepath)
            self.logger.info(f'++++++++> [BestCallback] Best Weights Saved at Epoch {epoch + 1} to {filepath}')
        else:
            if (self.mode == 'min' and val < self.best_val) or \
               (self.mode == 'max' and val > self.best_val):
                self.best_val = val
                self._save_model(epoch,filepath)
                self.logger.info(f'++++++++> [BestCallback] Best Weights Updated at Epoch {epoch + 1} to {filepath}')

    def _save_model(self, epoch, filepath):

        if self.save_weights_only:
            self.model.save_weights(filepath)
        else:
            self.model.save(filepath)

class LastEpochCallback(Callback):
    def __init__(self, logger, save_path, save_weights_only=True):

        super(LastEpochCallback, self).__init__()
        self.logger = logger
        self.save_path = save_path
        self.save_weights_only = save_weights_only
        self.last_epoch = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def on_epoch_end(self, epoch, logs=None):

        self.last_epoch = epoch

    def on_train_end(self, logs=None):

        filepath = os.path.join(self.save_path, f'model_epoch_{self.last_epoch + 1}.weights.h5')
        
        if self.save_weights_only:
            self.model.save_weights(filepath)
        else:
            self.model.save(filepath)
        
        self.logger.info(f'++++++++> [LastEpochCallback] Last Weights Saved at Epoch {self.last_epoch + 1} to {filepath}')
        
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

class Vgg16:
    def __init__(self, input_shape, n_classes, logger, type, first_trained_layer=None, save_dir='_weights'):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.first_trained_layer = first_trained_layer
        self.type = type
        self.model = self.build_model()
        self.save_dir = save_dir
        self.logger = logger

    def build_model(self):

        vgg16 = VGG16(include_top=False, input_shape=self.input_shape, weights='imagenet')

        # Freeze the specified first layers
        if self.first_trained_layer is not None:
            for layer in vgg16.layers[:self.first_trained_layer]:
                layer.trainable = False

        x = vgg16.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        if self.type['model']=='mc':
            x = MCDropout(0.5)(x)
        else:
            x = Dropout(0.5)(x)

        predictions = Dense(self.n_classes, activation='softmax')(x) 

        # Creating the final model
        model = Model(inputs=vgg16.input, outputs=predictions)

        return model

    def compile(self,
                learning_rate=1e-5,
                loss='categorical_crossentropy',
                metrics=['accuracy']):
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics)

    def fit(self, data, epochs, callbacks):

        self.model.fit(data,
                       epochs=epochs,
                       callbacks=[callbacks])

    def summary(self):
        self.model.summary()

    def save_weights(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Define the path for saving the weights
        weights_path = os.path.join(self.save_dir, 'vgg16_weights.weights.h5')
        
        # Save the model's weights
        self.model.save_weights(weights_path)
        self.logger.info(f'--------> Model Weights Saved To: {weights_path}')

    def load_weights(self, weights_path):

        self.model.load_weights(weights_path)
        self.logger.info(f'--------> Succesfully Loaded Weights at {weights_path} ...')

    def predict(self, input):
            
        if input.ndim == 3:
            input = np.expand_dims(input, axis=0)

        if self.type['model']=='normal':
            predictions = self.model.predict(input, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            pred_probs = np.max(predictions, axis=1)
            predictions = np.expand_dims(predictions, axis=1)
            return predictions, pred_classes, pred_probs 
        
        elif self.type['model']=='mc':
            n_iter = self.type['params']['iter']
            predictions = [self.model.predict(input, verbose=0) for _ in range(n_iter)]
            predictions = np.array(predictions)
            mean_predictions = np.mean(predictions, axis=0)
            pred_classes = np.argmax(mean_predictions, axis=1)
            pred_probs = np.max(mean_predictions, axis=1)
            predictions = np.transpose(predictions, (1, 0, 2))
            return predictions, pred_classes, pred_probs