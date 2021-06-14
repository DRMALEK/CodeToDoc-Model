import json
import utils
import tensorflow as tf
import re

def create_dataset(path):
    with open(path, "r") as input_file:
        apis = json.load(input_file)
        apis_code = []
        apis_doc = []
        for api in apis:
            doc_first_line = api["api_doc"].split('.', 1)[0]
            doc_first_line = re.sub('</\w+>', '', doc_first_line)
            if len(doc_first_line) > 0 and len(api["api_code"]) > 0:
                doc_first_line_cleaned = utils.preprocess_sentence((doc_first_line))
                api_code_cleaned = utils.preprocess_sentence(api['api_code'])
                if len(doc_first_line_cleaned) > 0 and len(api_code_cleaned) > 0: 
                    doc_first_line_cleaned = f'<start> {doc_first_line_cleaned} <end>' 
                    api_code_cleaned = f'<start> {api_code_cleaned} <end>'
                    apis_code.append(api_code_cleaned)
                    apis_doc.append(doc_first_line_cleaned)

                else:
                    continue                    
        assert len(apis_code) == len(apis_doc), "The lengths of comments and api seq is not equal !"
        return apis_code, apis_doc

def tokenize(lang, type=None):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='OVV')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    if type == 'target':           # For targets, do a turncations
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post', maxlen=30, value=0)
    else:
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post', value=0)

    return tensor, lang_tokenizer

def load_dataset(path):
    # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(path)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang, 'target')

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer