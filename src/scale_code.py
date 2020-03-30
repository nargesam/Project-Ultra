from typing import List, Tuple
from enigma.machine import EnigmaMachine
from faker import Faker
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


def predict(cipher_list: List[str]) -> List[str]:
    # solution here
    # uses part of the data to train
    # then uses the rest of the data to predict the cipher with .9  acc 
    predicted_plain = cipher_list
    # print(f" len cipher {len(predicted_plain)}")
    # print(print(predicted_plain[1:2]))

    # TODO: import test() from model.py and for each cipher in test predict and append to ret_list
    return predicted_plain


def str_score(str_a: str, str_b: str) -> float:
    if len(str_a) != len(str_b):
        return 0

    n_correct = 0

    for a, b in zip(str_a, str_b):
        n_correct += int(a == b)
    # print(f" n_correct {n_correct}")
    # print(f" len  {n_correct}")

    return n_correct / len(str_a)


def score(predicted_plain: List[str], correct_plain: List[str]) -> float:
    correct = 0

    for p, c in zip(predicted_plain, correct_plain):
        # print(p,c)
        # exit()
        if str_score(p, c) > 0.8:
            correct += 1
    print(f" correct {correct}")
    print(f" len correct_plain {len(correct_plain)}")

    return correct / len(correct_plain)


if __name__ == "__main__":
    plain, cipher = generate_data(1<<14)
    print(plain[0:3])
    print(cipher[0:3])

    print(len(plain), len(cipher))
    # print(type(plain), type(cipher))
    print(score(predict(cipher), plain))
