import os

from abc import ABC, abstractproperty, abstractmethod

import numpy as np
import pandas as pd

from keras.layers import Input, LSTM, Dense, GRU
from keras.models import Model
from keras.optimizers import Adam

import scale_code as sc


class EnigmaDataset:
    def __init__(self, n_samples, seq_len=42, save_file=None):
        self._n_samples = n_samples
        self._seq_len = seq_len
        self._save_file = save_file

        self._dataset = self._generate_data(
            n_samples=n_samples, 
            seq_len=seq_len, 
            save_file=save_file
        )

        self._train_data = None
        self._test_data = None
        self._train_data_partitioned = None
        self._test_data_partitioned = None

    @property
    def dataset(self):
        return self._dataset

    def _generate_data(self, n_samples, seq_len=42, save_file=None):
        # If dataset is not already created, generate dataset
        if save_file is not None and os.path.exists(save_file):
            enigma_data =  pd.read_csv(save_file, sep=",", header=0)

        else:
            plain, cipher = sc.generate_data(batch_size=n_samples, seq_len=seq_len)
            enigma_data = pd.DataFrame({'PLAIN': plain, 'CIPHER': cipher})

            if save_file:
                enigma_data.to_csv(save_file, index=False)

        return enigma_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def test_train_split(self, n_test_samples, sent_partition_size, random_state=42):
        # create test and train data
        df = self.dataset.copy()

        self._test_data = df.sample(
            n=n_test_samples, 
            replace=False, 
            random_state=random_state
        )
        
        self._train_data = df.drop(self._test_data.index)

        self._test_train_partition(sent_partition_size=sent_partition_size)

    @property
    def train_data_partitioned(self):
        return self._train_data_partitioned

    @property
    def test_data_partitioned(self):
        return self._test_data_partitioned
        
    def _divide_chunks(self, lst, n):
        # for each seq of chars, return list of sequences with size of n
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _test_train_partition(self, sent_partition_size):

        """ because 43 seq of chars is long, this function partitions
            every cipher/plain combo to at most len of sent_partition_size
            to predict smaller sequences
        """

        self._train_data_partitioned = self._partition_dataset(
            unpartitioned_dataset=self._train_data,
            sent_partition_size=sent_partition_size
        )

        self._test_data_partitioned = self._partition_dataset(
            unpartitioned_dataset=self._test_data,
            sent_partition_size=sent_partition_size
        )

    def _partition_dataset(self, unpartitioned_dataset, sent_partition_size):
        # create a new dataframe with partitioned sequences 
        df = {
            'ID': [], 
            'PLAIN': [], 
            'CIPHER': []
        }

        for index, row in unpartitioned_dataset.iterrows():    
            plain = list(str(row['PLAIN']))
            cipher = list(str(row['CIPHER']))
            
            plain = list(self._divide_chunks(lst=plain, n=sent_partition_size))
            cipher = list(self._divide_chunks(lst=cipher, n=sent_partition_size))
            
            for i in range(len(plain)):
                plain_now = "".join(plain[i])
                cipher_now = "".join(cipher[i])
                
                df['ID'].append(index)
                df['PLAIN'].append(plain_now)
                df['CIPHER'].append(cipher_now)
            
        df = pd.DataFrame(df)
        
        return df


class EncodedDataset:
    def __init__(self, unencoded_dataset): 
        """
        This class calls the PlainTextEncoding and creates four different objects 
        of the class. Using this class, you have access to 
        Plain and Cipher list of test and train. 

        unencoded_dataset: should be object of EnigmaDataset which gives us access to 
                            all partitioned trian/test. 
        """  
        self._unencoded_dataset = unencoded_dataset
        self._plain_train = self._get_plain_train()
        self._plain_test = self._get_plain_test()
        self._cipher_train = self._get_cipher_train()
        self._cipher_test = self._get_cipher_test()

    @property
    def plain_train(self):
        return self._plain_train

    def _get_plain_train(self):
        # get object of PlainTextEncoding and input the partitioned train_data for plain
        sents = self._unencoded_dataset.train_data_partitioned['PLAIN'].tolist()
        obj = PlainTextEncoding(sentences=sents)
        return obj

    @property
    def cipher_train(self):
        return self._cipher_train

    def _get_cipher_train(self):
        # get object of PlainTextEncoding and input the partitioned train_data for Cipher
        sents = self._unencoded_dataset.train_data_partitioned['CIPHER'].tolist()
        obj = CipherTextEncoding(sentences=sents)
        return obj

    @property
    def plain_test(self):
        return self._plain_test

    def _get_plain_test(self):
        # get object of PlainTextEncoding and input the partitioned test_data for Plain
        sents = self._unencoded_dataset.test_data_partitioned['PLAIN'].tolist()
        obj = PlainTextEncoding(sentences=sents)
        return obj

    @property
    def cipher_test(self):
        return self._cipher_test

    def _get_cipher_test(self):
        # get object of PlainTextEncoding and input the partitioned test_data for Cipher
        sents = self._unencoded_dataset.test_data_partitioned['CIPHER'].tolist()
        obj = CipherTextEncoding(sentences=sents)
        return obj


class TextEncodingBase(ABC):
    def __init__(self, sentences):

        """
        This is an abstract class which creates properties of plain-cipher lists for test-train. 
        The PlainTextEncoding subclass holds a method to create the one-hot vectors for plain and target. 
        The CipherTextEncoding subclass holds a method to create the one-hot vectors for cipher. 


        """
        self._sentences = sentences
        self._sentences_processed = None

        self._process_sentences()
        self._max_sentence_len = max([len(x) for x in self._sentences_processed])
        self._n_samples = len(self._sentences_processed)
        self._alphabet = self._get_alphabet()

        self._index_to_char_lookup = None
        self._char_to_index_lookup = None
        self._create_lookups()

        self._input_vector = None
        self._encode_one_hot_vectors()

    @property
    def sentences(self):
        return self._sentences

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def alphabet(self):
        return self._alphabet

    def _get_alphabet(self):
        # get the distinct alphabet used in sentences of sentences_processed
        alpha = set()
        
        for sent in self.sentences_processed:
            for ch in sent:
                if (ch not in alpha):
                    alpha.add(ch)
                    
        alpha = sorted(list(alpha))
        return alpha

    @property
    def sentences_processed(self):
        return self._sentences_processed

    @abstractmethod
    def _process_sentences(self):
        raise NotImplementedError()
    
    @property
    def max_sentence_len(self):
        return self._max_sentence_len

    @property
    def index_to_char_lookup(self):
        return self._index_to_char_lookup

    @property
    def char_to_index_lookup(self):
        return self._char_to_index_lookup

    def _create_lookups(self):
        # dictionary as a lookup table for index to character: 
        # Key is index and value is alphabet in alphabets
        self._index_to_char_lookup = {}

        # dictionary as a lookup table for index to character: 
        # Key is each char in alphabet and value is index
        self._char_to_index_lookup = {}

        for k, v in enumerate(self.alphabet):
            self._index_to_char_lookup[k] = v
            self._char_to_index_lookup[v] = k

    @property
    def input_vector(self):
        return self._input_vector

    @abstractmethod
    def _encode_one_hot_vectors(self):
        raise NotImplementedError()


class PlainTextEncoding(TextEncodingBase):
    def _process_sentences(self):
        # each plain sentence has '\t' as an indicator of the start of the sequence
        # each plain sentence has '\n' as an indicator of the end of the sequence
        self._sentences_processed = [f"\t{x}\n" for x in self._sentences]
        self._target_vector = None

    @property
    def target_vector(self):
        return self._target_vector

    def _encode_one_hot_vectors(self):
        # One-hot vectors with 3D dimention
        self._input_vector = np.zeros(
            shape=(self._n_samples, self._max_sentence_len, len(self.alphabet)), 
            dtype='float32'
        )
        
        # One-hot vectors with 3D dimention
        self._target_vector = np.zeros(
            shape=(self._n_samples, self._max_sentence_len, len(self.alphabet)), 
            dtype='float32'
        )

        for i in range(self._n_samples):                
            for k, ch in enumerate(self.sentences_processed[i]):
                self._input_vector[i, k, self._char_to_index_lookup[ch]] = 1

                # target sequence will be ahead by one char and will not include the starting char.
                if k > 0:
                    self._target_vector[i, k-1, self._char_to_index_lookup[ch]] = 1


class CipherTextEncoding(TextEncodingBase):
    def _process_sentences(self):
        self._sentences_processed = self._sentences

    def _encode_one_hot_vectors(self):
        self._input_vector = np.zeros(
            shape=(self._n_samples, self._max_sentence_len, len(self.alphabet)), 
            dtype='float32'
        )

        for i in range(self._n_samples):                
            for k, ch in enumerate(self.sentences_processed[i]):
                self._input_vector[i, k, self._char_to_index_lookup[ch]] = 1


class UltraCodeBreaker:
    def __init__(self, encoded_dataset):
        """
        This class creates LSTM encoder-decoder model to decode Cipher to Plain
        """
        self._encoded_dataset = encoded_dataset

        self._model = None
        self._encoder_model = None
        self._encoder_model_inf = None
        self._decoder_model = None
        self._decoder_model_inf = None

    @property
    def model(self):
        return self._model
    
    def _create_model(self, n_nodes=256):
        # Encoder model
        self._encoder_model = UltraEncoderModel(
            cipher_text=self._encoded_dataset.cipher_train,
            n_nodes=n_nodes
        )

        # Decoder model
        self._decoder_model = UltraDecoderModel(
            plain_text=self._encoded_dataset.plain_train,
            encoder_states=self._encoder_model.states,
            n_nodes=n_nodes
        )

        self._model = Model(
            inputs=[
                self._encoder_model.input, 
                self._decoder_model.input
            ],
            outputs=[self._decoder_model.output]
        )

    def train(self, n_nodes=256, batch_size=256, epochs=100, validation_split=0.3):
        # After creating the model, this LSTM sequential model will run for 100 epochs
        # Split 30% the training data for validation set 
        self._create_model(n_nodes=n_nodes)
        self._model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')

        self._model.fit(
            x=[
                self._encoded_dataset.cipher_train.input_vector, 
                self._encoded_dataset.plain_train.input_vector
            ], 
            y=self._encoded_dataset.plain_train.target_vector,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )

    def create_test_model(self, n_nodes=256):
        # Inference models for testing

        # Encoder inference model
        self._encoder_model_inf = Model(
            self._encoder_model.input, 
            self._encoder_model.states
        )

        # Decoder inference model
        decoder_state_input_h = Input(shape=(n_nodes,))
        decoder_state_input_c = Input(shape=(n_nodes,))
        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

        decoder_out, decoder_h, decoder_c = self._decoder_model.lstm(
            self._decoder_model.input, 
            initial_state=decoder_input_states
        )

        decoder_states = [decoder_h , decoder_c]
        decoder_out = self._decoder_model.dense(decoder_out)

        self._decoder_model_inf = Model(
            inputs=[self._decoder_model.input] + decoder_input_states,
            outputs=[decoder_out] + decoder_states
        )

    def _decode_seq(self, inp_seq):
        # Given an input sequence, this fuctions predicts the cipher to plain
        # Initial states value is coming from the encoder 
        states_val = self._encoder_model_inf.predict(inp_seq)
        
        target_seq = np.zeros((1, 1, len(self._encoded_dataset.plain_test.alphabet)))
        target_seq[0, 0, self._encoded_dataset.plain_test.char_to_index_lookup['\t']] = 1
        
        translated_sent = ''
        stop_condition = False
        
        while not stop_condition:
            decoder_out, decoder_h, decoder_c = self._decoder_model_inf.predict(x=[target_seq] + states_val)
            
            max_val_index = np.argmax(decoder_out[0,-1,:])
            sampled_char = self._encoded_dataset.plain_test.index_to_char_lookup[max_val_index]
            translated_sent += sampled_char
            
            # Stops if reached end of cipher or reached max len for given plain
            if ( (sampled_char == '\n') or (len(translated_sent) >  self._encoded_dataset.plain_test.max_sentence_len)) :
                stop_condition = True
            
            target_seq = np.zeros((1, 1, len(self._encoded_dataset.plain_test.alphabet)))
            target_seq[0, 0, max_val_index] = 1
            
            states_val = [decoder_h, decoder_c]
            
        return translated_sent

    def predict(self, save_file=None):
        # if predictions exists, read the prediction file, otherwise predict for all cipher instances
        if save_file is not None and os.path.exists(save_file):
            self.predictions = pd.read_csv(save_file, sep=",", header=0)
        else:
            predictions_partitioned = []

            for seq_index in range(self._encoded_dataset.cipher_test.n_samples):
                inp_seq = self._encoded_dataset.cipher_test.input_vector[seq_index:seq_index+1]
                translated_sent = self._decode_seq(inp_seq)
                predictions_partitioned.append(translated_sent)
            
            # add all predictions to the partitioned test_data
            self._encoded_dataset._unencoded_dataset.test_data_partitioned['DECRYPTED'] =\
                                                     [x.strip() for x in predictions_partitioned]
            
            if save_file:
                self._encoded_dataset._unencoded_dataset.test_data_partitioned.to_csv(save_file, index=False)

            self.predictions = self._encoded_dataset._unencoded_dataset.test_data_partitioned
            
    def evaluate(self, save_file=None):
        # merge all partitioned sequences by the index
        predictions_joined = self.predictions.groupby(['ID'], as_index=False, sort=False) \
            .agg(''.join)

        if save_file:
            predictions_joined.to_csv(save_file, index=False)

        # calculate the score
        evaluation_score = sc.score(list(predictions_joined['DECRYPTED']), list(predictions_joined['PLAIN']))

        print(f'This model achives evaluation acc of {evaluation_score}')


class UltraEncoderModel:
    def __init__(self, cipher_text, n_nodes=256):
        # encoder model class
        self._n_nodes = n_nodes

        encoder_input = Input(shape=(None, len(cipher_text.alphabet)))
        encoder_lstm = LSTM(n_nodes, return_state=True)
        encoder_outputs, encoder_h, encoder_c = encoder_lstm(encoder_input)
        encoder_states = [encoder_h, encoder_c]

        self._input = encoder_input
        self._states = encoder_states

    @property
    def input(self):
        return self._input

    @property
    def states(self):
        return self._states


class UltraDecoderModel:
    def __init__(self, plain_text, encoder_states, n_nodes=256):
        # decoder model class
        self._n_nodes = n_nodes

        decoder_input = Input(shape=(None, len(plain_text.alphabet)))
        decoder_lstm = LSTM(n_nodes, return_sequences=True, return_state=True)
        decoder_out, _ , _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(len(plain_text.alphabet), activation='softmax')
        decoder_out = decoder_dense(decoder_out)

        self._input = decoder_input
        self._output = decoder_out
        self._lstm = decoder_lstm
        self._dense = decoder_dense

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def lstm(self):
        return self._lstm

    @property
    def dense(self):
        return self._dense