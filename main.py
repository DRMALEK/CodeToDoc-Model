import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasetUtils import load_dataset

from TranslationModel import TranslationModel

if __name__ == "__main__":
    print("Loading the dataset  ...")
    # Try experimenting with the size of that dataset
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset("./dataset.json")
    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                      target_tensor,
                                                                                                    test_size=0.2)

    print(max_length_targ)
    print(max_length_inp)

    #Show length
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    #Parameters
    BUFFER_SIZE = len(input_tensor_train)
    batch_size = 64
    embedding_dim = 256
    units = 512
    epochs = 20
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,

                                                                reduction='none')

    print("Loading the model ...")
    model = TranslationModel(vocab_inp_size=vocab_inp_size, vocab_tar_size=vocab_tar_size,
                             units=units,
                             embedding_dim=embedding_dim,
                             optimizer=optimizer,
                             max_length_inp=max_length_inp,
                             max_length_targ=max_length_targ,
                             loss_object=loss_object,
                             batch_size=batch_size)

    print("Start training ...")
    model.train(input_tensor_train=input_tensor_train, target_tensor_train=target_tensor_train,
                epochs=epochs, target_lang=targ_lang)
