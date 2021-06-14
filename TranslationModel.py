import utils
import datasetUtils
import tensorflow as tf
import os
import time
from utils import loss_function
from ModelComponents import Encoder, Decoder
from utils import preprocess_sentence
import numpy as np
from metrics import Metrics


class TranslationModel:
    def __init__(self,
                 optimizer,
                 loss_object,
                 vocab_inp_size,
                 vocab_tar_size,
                 max_length_inp,
                 max_length_targ,
                 batch_size,
                 units=512,
                 embedding_dim=256,
                 epochs=20):

        self.batch_size = batch_size
        self.encoder = Encoder(
            vocab_inp_size, embedding_dim, units, batch_size)
        self.decoder = Decoder(
            vocab_tar_size, embedding_dim, units, batch_size)
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.units = units
        self.embedding_dim = embedding_dim
        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size
        self.max_length_inp = max_length_inp
        self.max_length_targ = max_length_targ
        self.epochs = epochs

    def step(self, inp, targ, enc_hidden, targ_lang, mode):
        loss = 0
        if mode == 'val':
            predicted_sequnces = np.zeros((self.batch_size, self.max_length_targ)) #(input_tensor_val * max_length_targ)
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder.call(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * self.batch_size, 1)
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                # predictions (batch_size * vocab)
                predictions, dec_hidden, _ = self.decoder.call(dec_input, dec_hidden, enc_output)
                loss += loss_function(self.loss_object, targ[:, t], predictions)
                if mode == 'train':
                    dec_input = tf.expand_dims(targ[:, t], 1)    # Teacher forcing - feeding the target as the next input
                else:
                    predicted_sequnces[:, t] = tf.math.argmax(input=predictions, axis=1).numpy() 
                    dec_input = tf.expand_dims(tf.math.argmax(input=predictions, axis=1), 1)
        
        batch_loss = (loss / int(targ.shape[1]))
        
        if mode == 'train':
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss
        else:
            return batch_loss, predictions


    def train(self, input_tensor_train, target_tensor_train, target_lang,
              input_tensor_val, target_tensor_val):
        buffer_size = len(input_tensor_train)
        steps_per_epoch = len(input_tensor_train) // self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)).shuffle(buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction='none')
        checkpoint_dir = './training_checkpoints'
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)
        manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=3)
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for epoch in range(self.epochs):
            print("Current Epoch ", epoch)
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.step(
                    inp, targ, enc_hidden, target_lang, mode='train')
                total_loss += batch_loss
             
                if batch % 100 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy():.4f}')
            # saving (checkpoint) the model every 2 epochs and do an evaluation
            if (epoch + 1) % 2 == 0:
                # do evaluation
                val_loss, rouge_score = self.evaluate(input_tensor_val, target_tensor_val, target_lang)
                print(f'validiaton loss: {val_loss}  \n rouge score {rouge_score}')
                manager.save()

            print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    def evaluate(self, input_tensor_val, target_tensor_val, targ_lang):
        predicted_sequnces = np.zeros((len(input_tensor_val), self.max_length_targ))
        loss = 0.
        buffer_size = len(input_tensor_val)
        steps_per_epoch = len(input_tensor_val) // self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        enc_hidden = self.encoder.initialize_hidden_state()
        refs = []
        preds = []
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                # preds (batch_size  * voab size * max)
                batch_loss, predicitons = self.step(inp, targ, enc_hidden, targ_lang, mode='val')
                loss += batch_loss
                
                print(predicitons[0])
                
                refs.extend(targ_lang.sequences_to_texts(targ.numpy()))
                preds.extend(targ_lang.sequences_to_texts(predicitons.numpy()))

                
        with open('./preds.txt', 'w+') as preds_out, open('./refs', 'w+') as refs_out:
            for ref in refs:
                refs_out.write(ref + '\n')
            for pred in preds:
                preds_out.write(pred + '\n')

        #blue_score = Metrics.calculate_bleu(candidates_path = './preds.txt', references_path='./refs')
        rouge_score = Metrics.calculate_rouge(candidates_path= './preds.txt', references_path='./refs')

        # take the average loss
        loss = loss / steps_per_epoch
        
        return loss, rouge_score

    def load_model(self):
        ckpt = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        manager = tf.train.CheckpointManager(
            ckpt, './training_checkpoints', max_to_keep=3)
        manager.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
