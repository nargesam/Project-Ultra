# a seq-to-seq model to decypher enigma machines
# The messages can be up to 42 characters in length

import tensorflow as tf
from typing import List, Tuple
from faker import Faker
import numpy  as  np
import pandas as pd

from enigma.machine import EnigmaMachine
from sklearn import model_selection

import re


class ConfiguredMachine:
    def __init__(self):
        self.machine = EnigmaMachine.from_key_sheet(
            rotors='II IV V',
            reflector='B',
            ring_settings=[1, 20, 11],
            plugboard_settings='AV BS CG DL FU HZ IN KM OW RX')

    def reset(self):
        self.machine.set_display('WXC')

    def encode(self, plain_str: str) -> str:
        self.reset()
        return self.machine.process_text(plain_str)

    def batch_encode(self, plain_list: List[str]) -> List[str]:
        encoded = list()
        for s in plain_list:
            encoded.append(self.encode(s))
        return encoded

def pre_process(input_str):
    return re.sub('[^a-zA-Z]', '', input_str).upper()

def generate_data(batch_size: int, seq_len: int = 42) -> Tuple[List[str], List[str]]:
    fake = Faker()
    machine = ConfiguredMachine()

    plain_list = fake.texts(nb_texts=batch_size, max_nb_chars=seq_len)
    plain_list = [pre_process(p) for p in plain_list]
    cipher_list = machine.batch_encode(plain_list)
    return plain_list, cipher_list



class Ultra():

    """
    create one-hot vectors of plain and cipher
    create on-hot vectors of chars 
    define the seq to seq model in keras 
    define evaluation metrics, compile and run

    """

    def __init__(self, plain, cipher):
        self.plain = plain
        self.cipher  = cipher

        self.size_plain = len(plain)
        self.size_cipher = len(cipher)

        self.maxlen_plain = len(max(plain,  key=len))
        self.maxlen_cipher = len(max(cipher, key=len))

    # def _test_set_generator(self):
    #     # idx  =  np.random.choice(range(self.size_plain), int(0.3*self.size_plain))
    #     self.plain_t, self.plain_test, self.cipher_t, self.cipher_test = model_selection.train_test_split(\
    #                                                                         self.plain, self.cipher, test_size =0.3,
    #                                                                         random_state=42)
    #     self.plain_train, self.plain_validation, self.cipher_train, self.cipher_validation = model_selection.train_test_split(\
    #                                                                         self.plain_t, self.cipher_t, test_size =0.2,
    #                                                                         random_state=42)

    def one_hot_vectors(self):
        
        self.chars_plain = set(''.join(self.plain))
        self.chars_cipher = set(''.join(self.cipher))
        # print(self.chars_cipher)
        # print(self.chars_plain)
        
        # initialize the on-hot vectors  with zeros
        self.one_hot_plain = np.zeros(shape=(self.size_plain, self.maxlen_plain, len(self.chars_plain)), dtype='float32')
        self.one_hot_cipher = np.zeros(shape=(self.size_cipher, self.maxlen_cipher, len(self.chars_cipher)), dtype='float32')
        self.target_one_hot = np.zeros(shape=(self.size_cipher, self.maxlen_plain, len(self.chars_plain)),dtype='float32')

        # dictionary of unique chars
        self.dict_char_int_plain = {char: index for index, char in enumerate(self.chars_plain)}
        self.dict_int_char_plain = {index: char for index, char in enumerate(self.chars_plain)}

        self.dict_char_int_cipher = {char:index for index, char in enumerate(self.chars_cipher)}

        # create one-hot vectors
        for i in range(self.size_plain):
            for k, ch in enumerate(self.plain[i]):
                self.one_hot_plain[i, k, self.dict_char_int_plain[ch]] = 1
                
                if k > 0:
                    self.target_one_hot[i, k-1, self.dict_char_int_plain[ch]] = 1
        
            for k, ch in enumerate(self.cipher[i]):
                self.one_hot_cipher[i, k, self.dict_char_int_cipher[ch]] = 1
        
        
    # def _test_set_generator(self):
    #     # idx  =  np.random.choice(range(self.size_plain), int(0.3*self.size_plain))
    #     self.plain_t, self.plain_test, self.cipher_t, self.cipher_test = model_selection.train_test_split(\
    #                                                                         self.one_hot_plain, self.one_hot_cipher, test_size =0.3,
    #                                                                         random_state=42)
    #     self.plain_train, self.plain_validation, self.cipher_train, self.cipher_validation = model_selection.train_test_split(\
    #                                                                         self.plain_t, self.cipher_t, test_size =0.2,
    #                                                                         random_state=42)

    #     # print(self.plain_train[2:4])

    def define_model(self):
        #encoder 
        self.encoder_input = tf.keras.layers.Input(shape=(None, len(self.chars_cipher)))
        encoder_LSTM = tf.keras.layers.LSTM(16, return_state=True)
        encoder_outputs, h_state, c_state = encoder_LSTM(self.encoder_input)
        self.encoder_states = [h_state, c_state]

        #decoder 
        self.decoder_input =  tf.keras.layers.Input(shape=(None, len(self.chars_plain)))
        self.decoder_LSTM = tf.keras.layers.LSTM(16, return_sequences=True, return_state=True)
        self.decoder_outputs, _ , _ = self.decoder_LSTM(self.decoder_input, initial_state=self.encoder_states)
        self.decoder_dense  = tf.keras.layers.Dense(len(self.chars_plain), activation='softmax')
        self.decoder_output  = self.decoder_dense(self.decoder_outputs)

        self.model = tf.keras.Model(inputs=[self.encoder_input,self.decoder_input], outputs=[self.decoder_output])
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')


    def train(self):
        # output the evaluation metric of the model
        self.model.fit(x=[self.one_hot_plain, self.one_hot_cipher], y=self.target_one_hot, \
                    batch_size=10, epochs=20, validation_split=0.3)



    def test_model(self):
        
        # write predict function on one example 

        # encoder for decipher
        self.encoder_model_predict = tf.keras.Model(self.encoder_input, self.encoder_states)

        # decoder for decipher
        decoder_state_input_c = tf.keras.layers.Input(shape=(16,))
        decoder_state_input_h = tf.keras.layers.Input(shape=(16,))

        decoder_o, decoder_h, decoder_c = self.decoder_LSTM(self.decoder_input, initial_state=[decoder_state_input_c,\
            decoder_state_input_h])
        
        decoder_states = [decoder_h, decoder_c]

        decoder_o = self.decoder_dense(decoder_o)
        self.decoder_model_predict = tf.keras.Model(inputs=[self.decoder_input]+ [decoder_state_input_h, decoder_state_input_c], \
                                                outputs=[decoder_o] + decoder_states)
        

    def decode_seq(self, inp_seq):
        #print('************', self.dict_char_int_plain)
        states_val = self.encoder_model_predict.predict(inp_seq)
    
        target_seq = np.zeros(shape=(1, 1, len(self.chars_plain)))
        # target_seq[0, 0, self.dict_char_int_cipher['\t']] = 1
        
        translated_sent = ''
        stop_condition = False
        
        while not stop_condition:
            
            decoder_out, decoder_h, decoder_c = self.decoder_model_predict.predict(x=[target_seq] + states_val)
            
            max_val_index = np.argmax(decoder_out[0,-1,:])
            sampled_decipher_char = self.dict_int_char_plain[max_val_index]
            translated_sent += sampled_decipher_char
            
            if (len(translated_sent) == len(inp_seq) or (len(translated_sent) > self.maxlen_plain)) :
                stop_condition = True
            
            target_seq = np.zeros(shape=(1, 1, len(self.chars_plain)))
            target_seq[0, 0, max_val_index] = 1
            
            states_val = [decoder_h, decoder_c]
            
        return translated_sent
    
    def test(self):
       # print(self.one_hot_plain[:, :, ])

        decipher_sent = self.decode_seq(self.one_hot_cipher[:, :,])
        print(type(self.one_hot_cipher))
        print(decipher_sent)



if __name__ == "__main__":
    plain, cipher = generate_data(1<<14)
    train_plain  = plain
    train_cipher = cipher
    if len(plain) != len(cipher):
        print("Enigma data generator is not correct! ")
        exit()
    # print('--------------------------')
    # print(plain)
    # print('--------------------------')
    # print(cipher)
    # print('--------------------------')

    # cnt = 0
    # for i in range(len(plain)):
    #     if len(plain[i]) != len(cipher[i]):
    #         cnt +=1

    # print(cnt)
    # exit()

    decipher = decipherEnigmaMachine(plain, cipher)

    decipher.one_hot_vectors()
    # decipher._test_set_generator()

    decipher.model()
    decipher.train()
    decipher.test_model()
    decipher.test()



    # print(type(plain), type(cipher))

    # print(cipher[:2], plain[:2])
    # print(f"max len cipher {len(max(cipher))}")
    # print(f"max len plain {len(max(plain))}")
    


    # print(len(plain), len(cipher))
    # print(type(plain), type(cipher))
    # print(score(predict(cipher), plain))
