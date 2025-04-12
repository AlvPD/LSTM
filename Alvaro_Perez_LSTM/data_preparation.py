import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import html
import re
import string
import pickle

class Preprocessing(object):
    def __init__(self, data_dir, stopwords_file=None, sequence_len=None, n_samples=None, test_size=0.2, val_samples=100, random_state=0, ensure_preprocessed=False):
        """
Inicializa una interfaz para cargar, preprocesar y dividir los datos en conjuntos de entrenamiento, validación y prueba.

Parámetros:
- data_dir: Directorio que contiene el archivo del dataset 'data.csv' con las columnas 'SentimentText' y 'Sentiment'
- stopwords_file: Opcional. Si se proporciona, elimina las palabras vacías (stopwords) de los datos originales
- sequence_len: Opcional. Sea m la longitud máxima de secuencia en el dataset. Se requiere que sequence_len >= m. Si es None, se asignará automáticamente a m
- n_samples: Opcional. Número de muestras a cargar del dataset (útil para datasets grandes). Si es None, se cargará todo el dataset (precaución: puede tardar en preprocesar si el dataset es muy grande)
- test_size: Opcional. Valor entre 0 y 1 que representa la proporción del dataset a incluir en el conjunto de prueba. Por defecto es 0.2
- val_samples: Opcional. Número absoluto de muestras de validación. Por defecto es 100
- random_state: Opcional. Semilla aleatoria para dividir los datos en conjuntos de entrenamiento, prueba y validación. Por defecto es 0
- ensure_preprocessed: Opcional. Si es True, verifica que el dataset ya esté preprocesado. Por defecto es False
        """
        self._stopwords_file = stopwords_file
        self._n_samples = n_samples
        self.sequence_len = sequence_len
        self._input_file = os.path.join(data_dir, 'data.csv')
        self._preprocessed_file = os.path.join(data_dir, "preprocessed_" + str(n_samples) + ".npz")
        self._vocab_file = os.path.join(data_dir, "vocab_" + str(n_samples) + ".pkl")
        self._tensors = None
        self._sentiments = None
        self._lengths = None
        self._vocab = None
        self.vocab_size = None

        # Prepara los datos
        if os.path.exists(self._preprocessed_file) and os.path.exists(self._vocab_file):
            print('Cargando archivos preprocesados ...')
            self.__load_preprocessed()
        else:
            if ensure_preprocessed:
                raise ValueError('Incapaz de encontrar archivos preprocesados.')
            print('Leyendo los datos ...')
            self.__preprocess()

        # Divide los datos en: entrenamiento, vadilación y prueba
        indices = np.arange(len(self._sentiments))
        x_tv, self._x_test, y_tv, self._y_test, tv_indices, test_indices = train_test_split(
            self._tensors,
            self._sentiments,
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self._sentiments[:, 0])
        self._x_train, self._x_val, self._y_train, self._y_val, train_indices, val_indices = train_test_split(
            x_tv,
            y_tv,
            tv_indices,
            test_size=val_samples,
            random_state=random_state,
            stratify=y_tv[:, 0])
        self._val_indices = val_indices
        self._test_indices = test_indices
        self._train_lengths = self._lengths[train_indices]
        self._val_lengths = self._lengths[val_indices]
        self._test_lengths = self._lengths[test_indices]
        self._current_index = 0
        self._epoch_completed = 0

    def __preprocess(self):
        """
Carga los datos desde data_dir/data.csv, preprocesa cada muestra cargada y almacena archivos intermedios para evitar reprocesamiento posterior.
        """
        # carga los datos
        data = pd.read_csv(self._input_file, nrows=self._n_samples)
        self._sentiments = np.squeeze(data.as_matrix(columns=['Sentiment']))
        self._sentiments = np.eye(2)[self._sentiments]
        samples = data.as_matrix(columns=['SentimentText'])[:, 0]

        # limpia el texto
        samples = self.__clean_samples(samples)

        # Perepara el vocabulario dict()
        vocab = dict()
        vocab[''] = (0, len(samples))  # añade una palabra vacía
        for sample in samples:
            sample_words = sample.split()
            for word in list(set(sample_words)):  # palabras distintas
                value = vocab.get(word)
                if value is None:
                    vocab[word] = (-1, 1)
                else:
                    encoding, count = value
                    vocab[word] = (-1, count + 1)

        # Remueve las palabras menos comunes

        sample_lengths = []
        tensors = []
        word_count = 1
        for sample in samples:
            sample_words = sample.split()
            encoded_sample = []
            for word in list(set(sample_words)):  # palabras distintas
                value = vocab.get(word)
                if value is not None:
                    encoding, count = value
                    if count / len(samples) > 0.0001:
                        if encoding == -1:
                            encoding = word_count
                            vocab[word] = (encoding, count)
                            word_count += 1
                        encoded_sample += [encoding]
                    else:
                        del vocab[word]
            tensors += [encoded_sample]
            sample_lengths += [len(encoded_sample)]

        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._lengths = np.array(sample_lengths)

        #
        self.sequence_len, self._tensors = self.__apply_to_zeros(tensors, self.sequence_len)

        # guarda archivos intermedios
        with open(self._vocab_file, 'wb') as f:
            pickle.dump(self._vocab, f)
        np.savez(self._preprocessed_file, tensors=self._tensors, lengths=self._lengths, sentiments=self._sentiments)

    def __load_preprocessed(self):
        """
       Carga archivos intermedios, evitando el preprocesamiento de datos
        """
        with open(self._vocab_file, 'rb') as f:
            self._vocab = pickle.load(f)
        self.vocab_size = len(self._vocab)
        load_dict = np.load(self._preprocessed_file)
        self._lengths = load_dict['lengths']
        self._tensors = load_dict['tensors']
        self._sentiments = load_dict['sentiments']
        self.sequence_len = len(self._tensors[0])

    def __clean_samples(self, samples):
        """
       Limpia las muestras.
      :param samples: Muestras a limpiar
      :return: muestras limpiadas
        """
        print('Limpiando muestras ...')
        # Prepara patrones regex
        ret = []
        reg_punct = '[' + re.escape(''.join(string.punctuation)) + ']'
        if self._stopwords_file is not None:
            stopwords = self.__read_stopwords()
            sw_pattern = re.compile(r'\b(' + '|'.join(stopwords) + r')\b')

        # limpia cada muestra
        for sample in samples:
            # Restaura caracteres HTML
            text = html.unescape(sample)

            # Remueve ususarios y URLs
            words = text.split()
            words = [word for word in words if not word.startswith('@') and not word.startswith('http://')]
            text = ' '.join(words)

            # Transforma a minúsculas
            text = text.lower()

            # Remueve signos de puntuación
            text = re.sub(reg_punct, ' ', text)

            # Remplaza caracteres repetidos por uno solo
            text = re.sub(r'([a-z])\1{2,}', r'\1', text)

            # Remueve las "stopwords"
            if stopwords is not None:
                text = sw_pattern.sub('', text)
            ret += [text]

        return ret

    def __apply_to_zeros(self, lst, sequence_len=None):
        """
        Rellena lst con ceros según sequence_len
        :param lst: Lista a rellenar
        :param sequence_len: Opcional. Sea m la longitud máxima de secuencia en lst. Se requiere que sequence_len >= m. Si sequence_len es None, se asignará automáticamente a m.
      :return: longitud de relleno utilizada y array numpy de tensores rellenados
        """
        # Encuentra la longitud máxima de m y asegura que m>=sequence_len
        inner_max_len = max(map(len, lst))
        if sequence_len is not None:
            if inner_max_len > sequence_len:
                raise Exception('Error: La longitud de secuencia proporcionada no es suficiente')
            else:
                inner_max_len = sequence_len

        # Pad list con ceros
        result = np.zeros([len(lst), inner_max_len], np.int32)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                result[i][j] = val
        return inner_max_len, result

    def __read_stopwords(self):
        """
        :return: Lista de stopwords (palabras vacías)
        """
        if self._stopwords_file is None:
            return None
        with open(self._stopwords_file, mode='r') as f:
            stopwords = f.read().splitlines()
        return stopwords

    def next_batch(self, batch_size):
        """
        :param batch_size: batch_size>0. Número de muestras que se incluirán
        :return: Devuelve muestras del tamaño del batch (text_tensor, text_target, text_length)
        """
        start = self._current_index
        self._current_index += batch_size
        if self._current_index > len(self._y_train):
            # Completa épocas y de manera aleatoria revuelve las muestras
            self._epoch_completed += 1
            ind = np.arange(len(self._y_train))
            np.random.shuffle(ind)
            self._x_train = self._x_train[ind]
            self._y_train = self._y_train[ind]
            self._train_lengths = self._train_lengths[ind]
            start = 0
            self._current_index = batch_size
        end = self._current_index
        return self._x_train[start:end], self._y_train[start:end], self._train_lengths[start:end]

    def get_val_data(self, original_text=False):
        """
        :param original_text: Opcional. Si se deben devolver las muestras originales o no
        :return: Devuelve los datos de validación. Si original_text es True, devuelve (original_samples, text_tensor, text_target, text_length), de lo contrario devuelve (text_tensor, text_target, text_length)
        """
        if original_text:
            data = pd.read_csv(self._input_file, nrows=self._n_samples)
            samples = data.as_matrix(columns=['SentimentText'])[:, 0]
            return samples[self._val_indices], self._x_val, self._y_val, self._val_lengths
        return self._x_val, self._y_val, self._val_lengths

    def get_test_data(self, original_text=False):
        """
        :param original_text: Opcional. Si se deben devolver las muestras originales o no
        :return: Devuelve los datos de prueba. Si original_text es True, devuelve (original_samples, text_tensor, text_target, text_length), de lo contrario devuelve (text_tensor, text_target, text_length)
        """
        if original_text:
            data = pd.read_csv(self._input_file, nrows=self._n_samples)
            samples = data[['SentimentText']].values[:, 0]
            return samples[self._test_indices], self._x_test, self._y_test, self._test_lengths
        return self._x_test, self._y_test, self._test_lengths
