import torch
import torch.nn as nn
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator
from spacy.lang.hi import Hindi
from spacy.lang.en import English
import os
import math
import random
import numpy as np
import torch.optim as optim
import time
import math
import pickle
import logging
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from torch.nn.utils.rnn import pack_padded_sequence
import sys


MODEL_NAME = 'vanilla-seq2seq'
CACHE_DIR = "/home/tushar/Desktop/MS/sem 2/nlpa/assignment-2/saved_models/vanilla"

smoothie = SmoothingFunction()
spacy_en, spacy_hi = English(), Hindi()
log_file = os.path.join(CACHE_DIR, "%s.log"%MODEL_NAME)
#logging to a file
logging.basicConfig(filename=os.path.abspath(log_file), filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
#logging to standard output
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



class EncoderRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, layer_count, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer_count, dropout=dropout_rate)
        
    def forward(self, input_data, src_field, device):
        #input dimensions : [seq_length, batch_size]
        embedding = self.dropout(self.embedding(input_data))
        #packing padded sequence
        packed_embedding = pack_padded_sequence(embedding, calculate_seq_length_in_batch(input_data, src_field, device), enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedding)
        
        #output dim = [seq_len, batch, hidden_dim]
        #hidden = [layer_count, batch, hidden_dim]
        #cell = [layer_count, batch, hidden_dim]
        
        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, layer_count, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer_count, dropout=dropout_rate)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, output_data, hidden_state, cell_state):
        #output dimensions : [batch_size]
        embedding = self.dropout(self.embedding(output_data.unsqueeze(0)))
        output, (next_hidden, next_cell) = self.rnn(embedding, (hidden_state, cell_state))
        
        #output dim = [1, batch, hidden_dim]
        #hidden dim = [layer_count, batch, hidden_dim]
        #cell dim = [layer_count, batch, hidden_dim]
        
        out_predict = self.out_layer(output.squeeze(0))
        return out_predict, next_hidden, next_cell

def extract_sents(reference_translation, predicted_translation, trg_field):
    # reference_translation dim : [seq_len, batch_size]
    # predicted_translation dim : [seq_len, batch_size, output_dimension]
    seq_length = reference_translation.shape[0]
    batch_size = reference_translation.shape[1]
    #initializing the words
    reference_sents, predicted_sents = [[] for x in range(batch_size)], [[] for x in range(batch_size)]
    #done[i][0] for reference and done[i][1] for predicted translation
    done = [[False, False] for x in range(batch_size)]
    eos_token = trg_field.eos_token
    ref_count, pred_count = 0, 0
    #find the max probability of the word at each time step
    predicted_translation = predicted_translation.argmax(2)
    for i in range(1, seq_length):
        if ref_count == batch_size and pred_count == batch_size:
            break
        for j in range(batch_size):
            #considering the reference translation
            if not done[j][0]:
                p_token = trg_field.vocab.itos[reference_translation[i, j]]
                if p_token == eos_token:
                    done[j][0] = True
                    ref_count += 1
                else:
                    reference_sents[j].append(p_token)
            #considering thr hypothesis translation
            if not done[j][1]:
                p_token = trg_field.vocab.itos[predicted_translation[i, j]]
                if p_token == eos_token:
                    done[j][1] = True
                    pred_count += 1
                else:
                    predicted_sents[j].append(p_token)
    return reference_sents, predicted_sents

def get_source_sentences(source, src_field):
    seq_length = source.shape[0]
    batch_size = source.shape[1]
    eos_token = src_field.eos_token
    src_count = 0
    src_sents = [[] for x in range(batch_size)]
    done = [False for x in range(batch_size)]
    for i in range(1, seq_length):
        if src_count == batch_size:
            break
        for j in range(batch_size):
            #considering the reference translation
            if not done[j]:
                p_token = src_field.vocab.itos[source[i, j]]
                if p_token == eos_token:
                    done[j] = True
                    src_count += 1
                else:
                    src_sents[j].append(p_token)
    return src_sents

class Seq2seqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2seqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src_batch, trg_batch, src_field, teacher_verse=0.8):
        #src_batch dim: [seq_len, batch]
        #trg_batch dim: [seq_len, batch]
        
        batch_size = src_batch.shape[1]
        trg_len = trg_batch.shape[0]
        #to store the output generated at each time step
        #out_pred dim : [seq_len, batch_size, out_dim]
        out_pred = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        
        hidden, cell = self.encoder(src_batch, src_field, self.device)
        
        decoder_input = trg_batch[0, :] #<sos>
        
        for i in range(1, trg_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            
            out_pred[i] = decoder_output
            
            teacher_verse_prob = random.random() < teacher_verse
            
            top1 = decoder_output.argmax(1)
            
            decoder_input = trg_batch[i, :] if teacher_verse_prob else top1.detach()
        
        return out_pred

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def store_object(pickle_object, file_path):
    try:
        with open(os.path.abspath(file_path), 'wb') as p_file:
            pickle.dump(pickle_object, p_file)
    except Exception as e:
        logging.info("[INFO] unable to store object to %s. Error : %s" % (file_path, str(e)))
        return False
    return True

def hindi_tokenizer(sentence):
    return [x.text for x in spacy_hi.tokenizer(sentence)]

def english_tokenizer(sentence):
    return [x.text for x in spacy_en.tokenizer(sentence)][::-1]

def print_dataset_statistics(train_data, valid_data, test_data, extension, fields):
    logging.info("[INFO] number of training examples : %s" % (len(train_data.examples)))
    logging.info("[INFO] number of validation examples : %s" % (len(valid_data.examples)))
    logging.info("[INFO] number of testing examples : %s" % (len(test_data.examples)))
    logging.info('--'*30)
    logging.info("[INFO] source language vocab (%s) : %s" % (extension[0], len(fields[0].vocab)))
    logging.info("[INFO] target language vocab (%s) : %s" % (extension[1], len(fields[1].vocab)))

def load_datasets(dataset_path, dataset_names, translate_pair, extentions, fields):
    final_datasets = []
    exts = [".%s"%x for x in extentions]
    for dataset_name in dataset_names:
        final_datasets.append(TranslationDataset(path=os.path.join(dataset_path, translate_pair, dataset_name), exts=exts, fields=[fields[0], fields[1]]))
    
    return final_datasets

def create_seq2seq_model(model_config, src_vocab, trg_vocab, device='cpu'):
    #encoder config
    enc_emb_dim = model_config['encoder']['emb_dim']
    enc_hid_dim = model_config['encoder']['hidden_dim']
    enc_layer_count = model_config['encoder']['layer_count']
    enc_dropout = model_config['encoder']['dropout']

    #decoder config
    dec_emb_dim = model_config['decoder']['emb_dim']
    dec_hid_dim = model_config['decoder']['hidden_dim']
    dec_layer_count = model_config['decoder']['layer_count']
    dec_dropout = model_config['decoder']['dropout']

    enc = EncoderRNN(src_vocab, enc_emb_dim, enc_hid_dim, enc_layer_count, enc_dropout)
    dec = DecoderRNN(trg_vocab, dec_emb_dim, dec_hid_dim, dec_layer_count, dec_dropout)

    return Seq2seqModel(enc, dec, device).to(device)

def train_model(model, iterator, optimizer, loss_function, clip, src_field, trg_field):
    #set the model in train mode so the dropout and other training parameter will be effective
    model.train()
    reference_sents, hypothesis_sents = [], []
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        output = model(src, trg, src_field)
        
        with torch.no_grad():
            new_refs, new_hypos = extract_sents(trg, output, trg_field)
            reference_sents.extend(new_refs)
            hypothesis_sents.extend(new_hypos)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = loss_function(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), corpus_bleu([[x] for x in reference_sents], hypothesis_sents)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate_model(model, iterator, loss_function, src_field, trg_field):
    model.eval()
    reference_sents, hypothesis_sents = [], []
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, src_field, 0) #turn off teacher forcing

            with torch.no_grad():
                new_refs, new_hypos = extract_sents(trg, output, trg_field)
                reference_sents.extend(new_refs)
                hypothesis_sents.extend(new_hypos)

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = loss_function(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), corpus_bleu([[x] for x in reference_sents], hypothesis_sents)

def translate(model, iterator, src_field, trg_field):
    model.eval()
    source_sents, reference_sents, hypothesis_sents = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, src_field, 0) #turn off teacher forcing

            with torch.no_grad():
                new_refs, new_hypos = extract_sents(trg, output, trg_field)
                new_src = get_source_sentences(src, src_field)
                source_sents.extend(new_src)
                reference_sents.extend(new_refs)
                hypothesis_sents.extend(new_hypos)
    return source_sents, reference_sents, hypothesis_sents

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_seq_length_in_batch(src_batch_tensor, src_field, device):
    seq_length = src_batch_tensor.shape[0]
    batch_size = src_batch_tensor.shape[1]
    length_vector = torch.zeros(batch_size).to(device)
    count = 0
    for i in range(seq_length-1, -1, -1):
        if count == batch_size:
            break
        for j in range(batch_size):
            if length_vector[j] == 0 and src_batch_tensor[i][j] == src_field.vocab.stoi[src_field.eos_token]:
                length_vector[j] = i
                count += 1
    return length_vector+1

def execute_training_loop(model, train_iterator, valid_iterator, loss_function, optimizer, clip_value, src_field, trg_field, epochs=3, model_cache_path='seq2seq-model.pt'):
    best_valid_loss = float('inf')
    stats = {
            "train" : [],
            "valid" : [],
        }
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_bleu = train_model(model, train_iterator, optimizer, loss_function, clip_value, src_field, trg_field)
        valid_loss, valid_bleu = evaluate_model(model, valid_iterator, loss_function, src_field, trg_field)
        stats["train"].append({'loss' : train_loss, 'bleu' : train_bleu})
        stats["valid"].append({'loss' : valid_loss, 'bleu' : valid_bleu})
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_cache_path)
        
        logging.info(f'[INFO] Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info(f'[INFO] \tTrain Loss: {train_loss:.3f} Train Bleu : {train_bleu:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logging.info(f'[INFO] \t Val. Loss: {valid_loss:.3f}  Val. Bleu : {valid_bleu:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    return stats

def init(model_config, device='cpu'):
    logging.critical("[CRITICAL] %s device is selected" % device)
    logging.info('[INFO] Using directory %s for the translation pair with filename %s' % (os.path.abspath(model_config['global']['dataset_path']), model_config['global']['translate_pair']))
    #initialize the field for src language
    src_field = Field(tokenize = english_tokenizer, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    #initialize the field for trg language
    trg_field = Field(tokenize = hindi_tokenizer, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    train_data, valid_data, test_data = load_datasets(model_config['global']['dataset_path'], model_config['global']['dataset_file_names'], model_config['global']['translate_pair'], model_config['global']['lang_extensions'], [src_field, trg_field])
    #initialize the vocabulary
    src_field.build_vocab(train_data, min_freq = 1)
    trg_field.build_vocab(train_data, min_freq = 1)
    #display dataset stats
    print_dataset_statistics(train_data, valid_data, test_data, model_config['global']['lang_extensions'], [src_field, trg_field])
    model = create_seq2seq_model(model_config, len(src_field.vocab), len(trg_field.vocab), device)
    optimizer = optim.Adam(model.parameters())
    #defining the loss function
    loss_function = nn.CrossEntropyLoss(ignore_index = trg_field.vocab.stoi[trg_field.pad_token])

    logging.info(model.apply(init_weights))
    logging.info('[INFO] Model has %s trainable parameters' % (count_parameters(model)))
    logging.info('[INFO] About to start the primary training loop')
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = model_config['global']['batch_size'], 
        device = device)
    cache_file_name = "%s-%s-%s-epoch-%s.pt" % (model_config['global']['name'], model_config['global']['lang_extensions'][0], model_config['global']['lang_extensions'][1], model_config['global']['epochs'])
    cache_file_path = os.path.join(model_config['global']['cache_path'], cache_file_name)
    stats = execute_training_loop(model, train_iterator, valid_iterator, loss_function, optimizer, model_config['global']['clip_value'], src_field, trg_field, epochs=model_config['global']['epochs'], model_cache_path=os.path.abspath(cache_file_path))
    
    stats_file_name = "%s-%s-%s-epoch-%s-stats.pickle" % (model_config['global']['name'], model_config['global']['lang_extensions'][0], model_config['global']['lang_extensions'][1], model_config['global']['epochs'])
    store_object(stats, os.path.join(model_config['global']['cache_path'], stats_file_name))

    logging.info("[INFO] loading the model %s" % (cache_file_name))
    model.load_state_dict(torch.load(os.path.abspath(cache_file_path)))
    test_loss, test_bleu = evaluate_model(model, test_iterator, loss_function, src_field, trg_field)
    logging.info(f'[INFO] | Test Loss: {test_loss:.3f} Test Bleu: {test_bleu:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == "__main__":
    done_training = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #seq2seq model configuration
    model_config = {
        'global' : {
            'name' : MODEL_NAME,
            'epochs' : 10,
            'clip_value' : 1,
            'batch_size' : 50,
            'cache_path': CACHE_DIR,
            'dataset_path' : "/home/tushar/Desktop/MS/sem 2/nlpa/assignment-2/data",
            #combination of <src><trg>
            "translate_pair" : "enghin",
            #in order of training, validation and testing 
            "dataset_file_names" : ['train', 'dev', 'test'],
            #<src> then <trg>
            'lang_extensions' : ['en', 'hi'],
        },
        'encoder' : {
            'emb_dim' : 256,
            'hidden_dim' : 512,
            'dropout' : 0.5,
            'layer_count' : 2,
        },
        'decoder' : {
            'emb_dim' : 256,
            'hidden_dim' : 512,
            'dropout' : 0.5,
            'layer_count' : 2,
        },
    }


    initial_seed = 1234
    random_seed(initial_seed)
    if not done_training:
        init(model_config, device)
    else:
        model_type = "vanilla"
        test_samples_count = 10 

        logging.critical("[CRITICAL] %s device is selected" % device)
        logging.info('[INFO] Using directory %s for the translation pair with filename %s' % (os.path.abspath(model_config['global']['dataset_path']), model_config['global']['translate_pair']))
        #initialize the field for src language
        src_field = Field(tokenize = english_tokenizer, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)
        #initialize the field for trg language
        trg_field = Field(tokenize = hindi_tokenizer, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)
        train_data, valid_data, test_data = load_datasets(model_config['global']['dataset_path'], model_config['global']['dataset_file_names'], model_config['global']['translate_pair'], model_config['global']['lang_extensions'], [src_field, trg_field])
        #initialize the vocabulary
        src_field.build_vocab(train_data, min_freq = 1)
        trg_field.build_vocab(train_data, min_freq = 1)
        #display dataset stats
        print_dataset_statistics(train_data, valid_data, test_data, model_config['global']['lang_extensions'], [src_field, trg_field])
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = model_config['global']['batch_size'], 
        device = device)

        cache_file_name = "%s-%s-%s-epoch-%s.pt" % (model_config['global']['name'], model_config['global']['lang_extensions'][0], model_config['global']['lang_extensions'][1], model_config['global']['epochs'])
        #model type used in cache_file_path
        cache_file_path = os.path.join(model_config['global']['cache_path'], model_type, cache_file_name)        
        model = create_seq2seq_model(model_config, len(src_field.vocab), len(trg_field.vocab), device)
        logging.info("[INFO] loading the model %s" % (cache_file_name))
        model.load_state_dict(torch.load(os.path.abspath(cache_file_path)))
        logging.info("[INFO] translating the test sentences")
        src_sents, ref_sents, hypo_sents = translate(model, test_iterator, src_field, trg_field)
        for i in range(test_samples_count):
            index = int(len(src_sents) * torch.rand(1).item())
            logging.info("source     : %s" % (' '.join(src_sents[index][::-1])))
            logging.info("reference  : %s" % (' '.join(ref_sents[index])))
            logging.info("predicited : %s" % (' '.join(hypo_sents[index])))
        logging.info("test bleu score : %s" % (corpus_bleu([[x] for x in ref_sents], hypo_sents, smoothing_function=smoothie.method3)))