from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import callbacks
import numpy as np
import pandas as pd

import enigma_model as em

nb_samples = 1<<16
enigma_data = pd.read_csv('./enigma_data.csv')
print(enigma_data.head())
print(enigma_data.shape)
test = enigma_data.sample(n=16384, replace=False, random_state=42)
enigma_data = enigma_data.drop(test.index)
print(len(test))
print(len(enigma_data))
nb_samples = 49152
nb_samples_test = 16384



# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
#  training data prep
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######


eng_sent = []
fra_sent = []
eng_chars = set()
fra_chars = set()
# nb_samples = enigma_data.shape[0]

# Process english and french sentences
for index, row in enigma_data.iterrows():
#     eng_line = str(lines[line]).split('\t')[0]
    eng_line = str(row['CIPHER'])
    
    # Append '\t' for start of the sentence and '\n' to signify end of the sentence
#     fra_line = '\t' + str(lines[line]).split('\t')[1] + '\n'
    fra_line = f"\t{str(row['PLAIN'])}\n"
    
    eng_sent.append(eng_line)
    fra_sent.append(fra_line)
    
    for ch in eng_line:
        if (ch not in eng_chars):
            eng_chars.add(ch)
            
    for ch in fra_line:
        if (ch not in fra_chars):
            fra_chars.add(ch)


fra_chars = sorted(list(fra_chars))
eng_chars = sorted(list(eng_chars))


print(eng_sent[0])
print(fra_sent[0].strip())
print(eng_chars)
print(fra_chars)


# dictionary to index each english character - key is index and value is english character
eng_index_to_char_dict = {}

# dictionary to get english character given its index - key is english character and value is index
eng_char_to_index_dict = {}

for k, v in enumerate(eng_chars):
    eng_index_to_char_dict[k] = v
    eng_char_to_index_dict[v] = k

# dictionary to index each french character - key is index and value is french character
fra_index_to_char_dict = {}

# dictionary to get french character given its index - key is french character and value is index
fra_char_to_index_dict = {}
for k, v in enumerate(fra_chars):
    fra_index_to_char_dict[k] = v
    fra_char_to_index_dict[v] = k

max_len_eng_sent = max([len(line) for line in eng_sent])
max_len_fra_sent = max([len(line) for line in fra_sent])

print(max_len_eng_sent)
print(max_len_fra_sent)

tokenized_eng_sentences = np.zeros(shape = (nb_samples,max_len_eng_sent,len(eng_chars)), dtype='float32')
tokenized_fra_sentences = np.zeros(shape = (nb_samples,max_len_fra_sent,len(fra_chars)), dtype='float32')
target_data = np.zeros((nb_samples, max_len_fra_sent, len(fra_chars)),dtype='float32')

# Vectorize the english and french sentences

for i in range(nb_samples):
    for k,ch in enumerate(eng_sent[i]):
        tokenized_eng_sentences[i,k,eng_char_to_index_dict[ch]] = 1
        
    for k,ch in enumerate(fra_sent[i]):
        tokenized_fra_sentences[i,k,fra_char_to_index_dict[ch]] = 1

        # decoder_target_data will be ahead by one timestep and will not include the start character.
        if k > 0:
            target_data[i,k-1,fra_char_to_index_dict[ch]] = 1


# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
#  models for training
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######



# Encoder model

encoder_input = Input(shape=(None,len(eng_chars)))
encoder_LSTM = LSTM(256,return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
encoder_states = [encoder_h, encoder_c]

# Decoder model

decoder_input = Input(shape=(None,len(fra_chars)))
decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(fra_chars),activation='softmax')
decoder_out = decoder_dense (decoder_out)


model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# filepath = './enigma_best_model_eng_to_cipher.h5'

# checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
# # reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
# earlystopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
# callbacks = [checkpoint]


model.fit(x=[tokenized_eng_sentences,tokenized_fra_sentences], 
          y=target_data,
          batch_size=64,
          epochs=130,
          validation_split=0.2)
        #   ,
        #   callbacks=callbacks)



# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# Inference models for testing
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######


# Encoder inference model
encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, 
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]

decoder_out = decoder_dense(decoder_out)

decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )

def decode_seq(inp_seq):
    
    # Initial states value is coming from the encoder 
    states_val = encoder_model_inf.predict(inp_seq)
    
    target_seq = np.zeros((1, 1, len(fra_chars)))
    target_seq[0, 0, fra_char_to_index_dict['\t']] = 1
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_fra_char = fra_index_to_char_dict[max_val_index]
        translated_sent += sampled_fra_char
        
        if ( (sampled_fra_char == '\n') or (len(translated_sent) > max_len_fra_sent)) :
            stop_condition = True
        
        target_seq = np.zeros((1, 1, len(fra_chars)))
        target_seq[0, 0, max_val_index] = 1
        
        states_val = [decoder_h, decoder_c]
        
    return translated_sent

# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# Create test on-hot encoder 
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######


nb_samples_test = 16384
eng_sent_test = []
fra_sent_test = []
eng_chars_test = set()
fra_chars_test = set()
# nb_samples = enigma_data.shape[0]

# Process english and french sentences
for index, row in test.iterrows():
#     eng_line = str(lines[line]).split('\t')[0]
    eng_line = str(row['CIPHER'])
    
    # Append '\t' for start of the sentence and '\n' to signify end of the sentence
#     fra_line = '\t' + str(lines[line]).split('\t')[1] + '\n'
    fra_line = f"\t{str(row['PLAIN'])}\n"
    
    eng_sent_test.append(eng_line)
    fra_sent_test.append(fra_line)
    
    for ch in eng_line:
        if (ch not in eng_chars_test):
            eng_chars_test.add(ch)
            
    for ch in fra_line:
        if (ch not in fra_chars_test):
            fra_chars_test.add(ch)
fra_chars_test = sorted(list(fra_chars_test))
eng_chars_test = sorted(list(eng_chars_test))

# dictionary to index each english character - key is index and value is english character
eng_index_to_char_dict_test = {}

# dictionary to get english character given its index - key is english character and value is index
eng_char_to_index_dict_test = {}

for k, v in enumerate(eng_chars_test):
    eng_index_to_char_dict_test[k] = v
    eng_char_to_index_dict_test[v] = k

max_len_eng_sent_test = max([len(line) for line in eng_sent_test])
max_len_fra_sent_test = max([len(line) for line in fra_sent_test])



tokenized_eng_sentences_test = np.zeros(shape = (nb_samples_test,max_len_eng_sent_test,len(eng_chars_test)), dtype='float32')
tokenized_fra_sentences_test = np.zeros(shape = (nb_samples_test,max_len_fra_sent_test,len(fra_chars_test)), dtype='float32')
target_data_test = np.zeros((nb_samples_test, max_len_fra_sent_test, len(fra_chars_test)),dtype='float32')


# Vectorize the english and french sentences

for i in range(nb_samples_test):
    for k,ch in enumerate(eng_sent_test[i]):
        tokenized_eng_sentences_test[i,k,eng_char_to_index_dict_test[ch]] = 1
        

# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
#  test on the data 
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######
# ######## ######## ######## ######## ######## ######## ######## ######## ######## #######

actual_english = []
predicted_cipher = []
actual_cipher = []


cipher = list(test['PLAIN'])
eng = list(test['CIPHER'])

for seq_index in range(nb_samples_test):
    inp_seq = tokenized_eng_sentences_test[seq_index:seq_index+1]
    actual_english.append(eng[seq_index])
    translated_sent = decode_seq(inp_seq)
    predicted_cipher.append(translated_sent)
    actual_cipher.append(cipher[seq_index])
    
    # print('-' * 100)
    # print('Input sentence:', eng_sent_test[seq_index])
    # print('Decoded sentence:', translated_sent.strip())
    # print('Decoded original:', cipher[seq_index])

   
    
def str_score(str_a, str_b) :
#     if len(str_a) != len(str_b):
#         return 0

    n_correct = 0

    for a, b in zip(str_a, str_b):
        n_correct += int(a == b)
    # print(f" n_correct {n_correct}")
    # print(f" len  {n_correct}")

    return n_correct / len(str_a)

    
def score(predicted_plain, correct_plain):
    correct = 0

    for p, c in zip(predicted_plain, correct_plain):
        # print(p,c)
        # exit()
        if str_score(p, c) > 0.8:
            correct += 1
    # print(f" correct {correct}")
    # print(f" len correct_plain {len(correct_plain)}")

    return correct / len(correct_plain)

# print(actual_english)
# print(predicted_cipher)
# print(actual_cipher)

test_result = pd.DataFrame()
test_result['PLAIN'] = actual_english
test_result['CIPHER'] = actual_cipher
test_result['PRED_CIPHER'] = predicted_cipher

test_result.to_csv('./test_result_256_130epoch_noCallBacks_Cipher_to_Eng.csv')

print(score(predicted_cipher, actual_cipher ))





# predicted_cipher = []
# actual_cipher = []

# for seq_index in range(1):
#     inp_seq = tokenized_eng_sentences[seq_index:seq_index+1]
#     translated_sent = decode_seq(inp_seq)
#     predicted_cipher.append(translated_sent)
#     actual_cipher.append(enigma_data['CIPHER'][seq_index])
    
# #     print('-' * 100)
# #     print('Input sentence:', eng_sent[seq_index])
# #     print('Decoded sentence:', translated_sent.strip())
# #     print('Decoded original:', enigma_data['CIPHER'][seq_index])

   
    
# def str_score(str_a, str_b) :
#     # if len(str_a) != len(str_b):
#     #     return 0

#     n_correct = 0

#     for a, b in zip(str_a, str_b):
#         print(a,b)
#         n_correct += int(a == b)
#     # print(f" n_correct {n_correct}")
#     # print(f" len  {n_correct}")

#     return n_correct / len(str_a)

    
# def score(predicted_plain, correct_plain):
#     correct = 0

#     for p, c in zip(predicted_plain, correct_plain):
#         # print(p,c)
#         # exit()
#         s = str_score(p, c)
#         if s > 0.8:
#             correct += 1
#     # print(f" correct {correct}")
#     # print(f" len correct_plain {len(correct_plain)}")

#     return correct / len(correct_plain)


# print(predicted_cipher)
# print(actual_cipher)


# print(score(predicted_cipher, actual_cipher ))





# # scores = model.evaluate(x=[encoder_input, decoder_input],y=[decoder_out], verbose=0)
# # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# # # serialize model to JSON
# # model_json = model.to_json()
# # with open("model.json", "w") as json_file:
# #     json_file.write(model_json)
# # # serialize weights to HDF5
# # model.save_weights("model.h5")
# # print("Saved model to disk")
 
# # # # later...
 
# # load json and create model
# # json_file = open('model.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # loaded_model = model_from_json(loaded_model_json)
# # # load weights into new model
# # loaded_model.load_weights("enigma_best_model.h5")
# # print("Loaded model from disk")
 
# # # evaluate loaded model on test data
# # loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# # score = loaded_model.evaluate(X, Y, verbose=0)
# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))