# From https://github.com/abisee/pointer-generator changed s.t. we use ower preprocessing

import sys
import os
import io
import pickle
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

from pathlib import Path

from src.preprocessing import load_verdict

DATA_PATH = Path("data/dataset")
STORAGE_PATH = Path("data/binary")
CHUNKS_PATH = STORAGE_PATH/"chunks"
MODEL_PATH = Path("model")

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

chunks_dir = STORAGE_PATH/"chunked"

VOCAB_SIZE = 50000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def get_train_files():
    """ Returns all the files previously selected to be used for training. """
    with io.open(MODEL_PATH/"train_files.pkl", "rb") as f:
        return pickle.load(f)

def get_val_files():
    """ Returns all the files previously selected to be used for validation. """
    with io.open(MODEL_PATH/"val_files.pkl", "rb") as f:
        return pickle.load(f)

def get_test_files():
    """ Returns all the files previously selected to be used for testing. """
    with io.open(MODEL_PATH/"test_files.pkl", "rb") as f:
        return pickle.load(f)


def chunk_file(set_name):
    in_file = STORAGE_PATH/('%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = CHUNKS_PATH/('%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
   # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in CHUNKS_PATH")

def get_art_abs(file):
    # We get the tokenized outputs for each segment
    verdict = load_verdict(file, normalize=True)

    # Make article into a single string
    text_lines = [" ".join(s) for s in verdict["facts"] + verdict["reasoning"]]
    article = " ".join(text_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abs_lines = [" ".join(s) for s in verdict["guiding_principle"]]
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abs_lines])

    return article, abstract


def write_to_bin(files, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    print("WRITING")
    num_stories = len(files)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx,s in enumerate(files):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))


            # Get the strings to write to .bin file
            article, abstract = get_art_abs(s)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with io.open(STORAGE_PATH/"vocab", 'w', encoding="utf-8") as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

if __name__ == '__main__':
    # Read the tokenized stories, do a little postprocessing then write to bin files
    """write_to_bin(get_test_files(), STORAGE_PATH/"test.bin")
    write_to_bin(get_val_files(), STORAGE_PATH/"val.bin")
    write_to_bin(get_train_files(), STORAGE_PATH/"train.bin", makevocab=True)"""

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
