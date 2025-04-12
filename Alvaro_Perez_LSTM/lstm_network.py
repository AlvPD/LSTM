# neural_network.py: Modelo LSTM en TensorFlow para Análisis de Sentimientos
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
import os

# Configuración de advertencias y logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LSTM_RNN_Network(Model):
    def __init__(self, hidden_size, vocab_size, embedding_size, max_length, n_classes=2, learning_rate=0.01, random_state=None):
        """
        Construye un modelo LSTM para análisis de sentimientos
        
        Args:
            hidden_size: Lista con el número de unidades en cada capa LSTM
            vocab_size: Tamaño del vocabulario (número de palabras únicas)
            embedding_size: Dimensión del vector de embedding para cada palabra
            max_length: Longitud máxima de las secuencias de entrada
            n_classes: Número de clases de salida (default: 2 para positivo/negativo)
            learning_rate: Tasa de aprendizaje para el optimizador (default: 0.01)
            random_state: Semilla para reproducibilidad (default: None)
        """
        super(LSTM_RNN_Network, self).__init__()
        
        # Configuración de parámetros
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Capa de embedding
        self.embedding = Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_size, 
            input_length=max_length
        )
        
        # Capas LSTM
        self.lstm_layers = []
        for i, units in enumerate(hidden_size):
            return_sequences = (i < len(hidden_size)-1)  # Solo retornar secuencias si no es la última capa
            self.lstm_layers.append(
                LSTM(
                    units, 
                    return_sequences=return_sequences,
                    dropout=0.5,  # Dropout para regularización
                    recurrent_dropout=0.5,  # Dropout recurrente
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=random_state)
                )
            )
        
        # Capa de salida
        self.output_layer = Dense(
            n_classes, 
            activation='softmax',
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=random_state)
        )
        
        # Optimizador
        self.optimizer = RMSprop(learning_rate=learning_rate)
        
        # Métricas
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.accuracy_metric = tf.keras.metrics.Accuracy(name='accuracy')
        self.precision_metric = tf.keras.metrics.Precision(name='precision')
        self.recall_metric = tf.keras.metrics.Recall(name='recall')

    def call(self, inputs, training=False):
        """
        Paso forward del modelo
        
        Args:
            inputs: Tensor de entrada con forma [batch_size, max_length]
            training: Booleano que indica si está en modo entrenamiento (afecta dropout)
            
        Returns:
            Predicciones del modelo con forma [batch_size, n_classes]
        """
        # Capa de embedding
        x = self.embedding(inputs)
        
        # Capas LSTM
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        
        # Capa de salida
        return self.output_layer(x)

    def train_step(self, data):
        """
        Paso de entrenamiento personalizado
        
        Args:
            data: Tupla de (entradas, etiquetas)
            
        Returns:
            Diccionario con las métricas calculadas
        """
        inputs, targets = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(inputs, training=True)
            
            # Cálculo de pérdida
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    targets, predictions, from_logits=False)
            )
        
        # Cálculo de gradientes y actualización de pesos
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Actualización de métricas
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(
            tf.argmax(targets, axis=1), 
            tf.argmax(predictions, axis=1)
        )
        self.precision_metric.update_state(
            tf.argmax(targets, axis=1),
            tf.argmax(predictions, axis=1)
        )
        self.recall_metric.update_state(
            tf.argmax(targets, axis=1),
            tf.argmax(predictions, axis=1)
        )
        
        return {
            'loss': self.loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'precision': self.precision_metric.result(),
            'recall': self.recall_metric.result()
        }

    def test_step(self, data):
        """
        Paso de evaluación personalizado
        
        Args:
            data: Tupla de (entradas, etiquetas)
            
        Returns:
            Diccionario con las métricas calculadas
        """
        inputs, targets = data
        
        # Forward pass (sin dropout)
        predictions = self(inputs, training=False)
        
        # Cálculo de pérdida
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                targets, predictions, from_logits=False)
        )
        
        # Actualización de métricas
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(
            tf.argmax(targets, axis=1), 
            tf.argmax(predictions, axis=1)
        )
        self.precision_metric.update_state(
            tf.argmax(targets, axis=1),
            tf.argmax(predictions, axis=1)
        )
        self.recall_metric.update_state(
            tf.argmax(targets, axis=1),
            tf.argmax(predictions, axis=1)
        )
        
        return {
            'loss': self.loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'precision': self.precision_metric.result(),
            'recall': self.recall_metric.result()
        }

    def reset_metrics(self):
        """Reinicia todas las métricas del modelo"""
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()

    def get_metrics(self):
        """
        Obtiene las métricas actuales del modelo
        
        Returns:
            Diccionario con las métricas actuales
        """
        return {
            'loss': self.loss_metric.result(),
            'accuracy': self.accuracy_metric.result(),
            'precision': self.precision_metric.result(),
            'recall': self.recall_metric.result()
        }

    def compile_model(self):
        """Configura el modelo para entrenamiento"""
        self.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )