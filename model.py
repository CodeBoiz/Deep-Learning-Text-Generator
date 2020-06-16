#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   The purpose of this python script is to both experiment with new deep learning techniques,
   and to further my knowledge on the subject of deep learning.

   I used this as a reference guide:
   https://www.tensorflow.org/tutorials/text/text_generation
"""

# import neccessary libaries
import tensorflow as tf
import numpy as np
import os
import time

# these are for generating a random logs folder
import string
import random

# root directory of the project
root_path = os.path.dirname(os.path.realpath(__file__))

############################################################
#  Create Dataset
############################################################

def create_dataset():
  # path to dataset
  dataset_path = os.path.join(root_path, r"dataset/Harry Potter and the Sorcerer.txt")

  # Read, then decode for py2 compat.
  text = open(dataset_path, 'rb').read().decode(encoding='unicode_escape')
  # length of text is the number of characters in it

  # The unique characters in the file
  vocab = sorted(set(text))

  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text_as_int = np.array([char2idx[c] for c in text])

  # The maximum length sentence we want for a single input in characters
  seq_length = 100
  examples_per_epoch = len(text)//(seq_length+1)

  # Create training examples / targets
  char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

  # Create the sequence from the dataset
  sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
   
  # Split the dataset up for better readability and passing into epochs and batch sizes
  dataset = sequences.map(split_input_target)

  return dataset, vocab, idx2char, char2idx

############################################################
#  Settings
############################################################
# Initalization of loss value as global variable
# to be used in multiple functions
loss = 0

# Number of RNN units
rnn_units = 1024

# The embedding dimension
embedding_dim = 256
# Batch size
BATCH_SIZE = 64

# Number of epochs to run through
EPOCHS=10

############################################################
#  Model
############################################################

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def train_model(dataset, vocab):
  # Buffer size to shuffle the dataset
  # (TF data is designed to work with possibly infinite sequences,
  # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
  # it maintains a buffer in which it shuffles elements).
  BUFFER_SIZE = 10000
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  # Length of the vocabulary in chars
  vocab_size = len(vocab)

  model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  #print out the different layors of the model
  model.summary()

  sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

  sampled_indices

  def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

  example_batch_loss  = loss(target_example_batch, example_batch_predictions)
  print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
  print("scalar_loss:      ", example_batch_loss.numpy().mean())

  model.compile(optimizer='adam', loss=loss)

  # generate a random string 15 characters long
  random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(15))

  # Directory where the checkpoints will be saved
  checkpoint_dir = './logs/' + random_string + "/"

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

  # Create a path for the saving location of the model
  model_dir = checkpoint_dir + "model.h5"

  # Save the model
  # TODO: Known issue with saving the model and loading it back
  # later in the script causes issues. Working to fix this issue
  model.save(model_dir)

  # Train from the last checkpoint
  tf.train.latest_checkpoint(checkpoint_dir)

  # Build the model with the dataset generated earlier
  model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

  # Load the weights
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

  # Build the model
  model.build(tf.TensorShape([1, None]))

  # Print out the model summary
  model.summary()

  # Return the model
  return model

############################################################
#  Generate Text
############################################################

def generate_text(model, idx2char, char2idx, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

############################################################
#  Configure
############################################################

if __name__ == '__main__':
    import argparse
    from keras.models import load_model

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'generate'")
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights",
                        help="Path to weights file")
    parser.add_argument('--start', required=False,
                        metavar="start of string",
                        help="The word that will begin the output string")

    args = parser.parse_args()


    # Configurations
    if args.command == "train":
      #Load in the dataset and other function to use
      dataset, vocab, idx2char, char2idx = create_dataset()
      #Train the model
      model = train_model(dataset, vocab)
      print(generate_text(model, idx2char, char2idx,start_string=u"Harry"))
    
    if args.command == 'generate':
      #Load in the dataset and other function to use
      dataset, vocab, idx2char, char2idx = create_dataset()
      # Get the model path
      model_path = os.path.join(root_path, args.weights)
      # Length of the vocabulary in chars
      vocab_size = len(vocab)
      # Build the model with the dataset generated earlier
      model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
      # Load the weights
      model.load_weights(model_path)
      # Build the model
      model.build(tf.TensorShape([1, None]))
      # Generate text
      print(generate_text(model, idx2char, char2idx,start_string=u"Harry"))
