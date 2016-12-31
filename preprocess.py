from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from core.data import data_utils

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default="")

    p.add_argument('--train_src', type=str, default="")
    p.add_argument('--train_tgt', type=str, default="")
    p.add_argument('--valid_src', type=str, default="")
    p.add_argument('--valid_tgt', type=str, default="")

    p.add_argument('--save_data', type=str, default="")

    p.add_argument('--src_vocab_size', type=int, default=50000)
    p.add_argument('--tgt_vocab_size', type=int, default=50000)
    p.add_argument('--src_vocab', type=str, default="data")
    p.add_argument('--tgt_vocab', type=str, default="save")
    p.add_argument('--features_vocabs_prefix', type=str, default="save")

    p.add_argument('--src_seq_length', type=int, default=50)
    p.add_argument('--tgt_seq_length', type=int, default=50)
    p.add_argument('--shuffle', type=int, default=1)
    p.add_argument('--seed', type=int, default=3435)

    p.add_argument('--report_every', type=int, default=100000)
    args = p.parse_args()

    preprocess(args)

def preprocess(args):
    print("preparing the custom data...")
    enc_train, dec_train, enc_dev, dec_dev, a, b = data_utils.prepare_custom_data(
        'working_dir', 'data/train.enc', 'data/train.dec',
        'data/test.enc', 'data/test.dec',
        10000,10000)
    print("enc_train", enc_train)
    print("dec_train", dec_train)
    print("enc_dev", enc_dev)
    print("dec_dev", dec_dev)
    print("a", a)
    print("b", b)

if __name__ == "__main__":
    main()
