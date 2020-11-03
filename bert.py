"""BERT NER Inference."""

from __future__ import absolute_import, division, print_function

import json
import os
import time
import torch
from tabulate import tabulate
from sys import getsizeof
import torch.nn.functional as F
from nltk import word_tokenize
from pytorch_transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)

GLOBAL_DEVICE = 'cpu'

class BertNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32, device=GLOBAL_DEVICE if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class Ner:

    def __init__(self,model_dir: str):
        self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config["label_map"]
        self.max_seq_length = self.model_config["max_seq_length"]
        self.label_map = {int(k):v for k,v in self.label_map.items()}
        self.device = GLOBAL_DEVICE if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            if len(tokens) + len(token) < self.model.bert.config.max_position_embeddings:
                tokens.extend(token)
                for i in range(len(token)):
                    if i == 0:
                        valid_positions.append(1)
                    else:
                        valid_positions.append(0)
            else:
                words = words[:i]
                break
        return tokens, valid_positions, words

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions, words = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions, words

    def predict(self, text: str, verbose = True):
        input_ids,input_mask,segment_ids,valid_ids, words = self.preprocess(text)
        input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
        input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
        segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
        valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
        with torch.no_grad():
            t = time.time()
            logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
            if verbose: print(f'inference times: {time.time()-t:0.2f} seconds')

        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

        logits = []
        pos = 0
        for index,mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos],logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label],confidence) for label,confidence in logits]
        # words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [(word,label,confidence) for word,(label,confidence) in zip(words,labels) if label not in ['O','[SEP]']]
        return output


model = Ner("out_base/")

#%%
input_seq = "Dr. Franklin provides us a chance to celebrate Thanksgiving, the Italian way, offering delicious food, wine, music, and dancing.  \
                Jackson is in love with Elizabeth.  \
                Richard works at VMWare, Georgia. Jackson is from Elizabeh  , New York.     \
                Bob and Alice are very close friends. They are also secretely in love! This is a very well kept secret in San Francisco but Tyler knows    \
                Dr. Leo Mirani from New York loves to shop at Target"

print('')
print('--------------------------------')
start = time.time()
ner_map = model.predict(input_seq)
print(f'total time: {time.time()-start:.2f} seconds')
print('--------------------------------')
print(tabulate(ner_map, headers=['Word', 'NER', 'Confidence']))
print('--------------------------------')

#%%
# input_seq = "Austria's Chancellor, Sebastian said that the victims included \
# \"an elderly man, an elderly woman, a young passer-by and a waitress.\" In  \
# addition to the four civilians, a gunman was shot dead by police. \
# Authorities have identified the attacker as Fejzulai Kujtim, a 20-year-old \
# Austrian man from 33 miles west of Vienna. Kujtim was a supporter of Islamic State, \
# who was sentenced to 22 months in prison on April 25, 2019 for attempting to travel \
# to Syria to join ISIS, Minister Karl Nehammer told state news agency APA. On December 5, \
# he was released early on parole, it reports. Police in Vienna have arrested 14 people \
# and searched 18 properties in relation to the attack. Initial reports on Monday night \
# said multiple gunmen opened fire at six locations in the city center, as residents savored \
#  the final hours of freedom before the imposition of a nationwide lockdown. \
# But authorities have since cast doubt on whether the man police shot was part \
# of a larger group. Austrian police said Tuesday morning \"they assume that there  \
# were more attackers\" said at the press conference, \"it can't be excluded that \
# there were more attackers.\" Austrian authorities told CNN they cannot rule \
# if a second suspect is on the run. Vienna police spokesperson Christopher \
# Verhnjak said police had been informed by witnesses there could be more than one \
# attacker. Police are investigating and advising people to stay home until they are \
# sure there isn't a suspect in hiding. Armed forces have been deployed in Vienna to \
# help secure the situation, with authorities indicating earlier in the evening that \
# at least one gunman remains on the loose. Residents of Vienna have been asked to stay \
# at home or in a safe place and follow the news. Authorities have abandoned compulsory \
# school attendance and asked citizens to avoid the city center for fears of another attacker \
# still at large. Earlier, Vienna police said that SWAT teams entered the gunman's apartment \
# using explosives and a search of its surroundings was underway. Police have also received \
# more than 20,000 videos from members of the public following the attack The initial \
# attack, which began around 8 p.m., was centered on the busy shopping and dining \
# district near Vienna's main synagogue, Seitenstettengasse Temple, which was closed. \
# The five other locations were identified as Salzgries, Fleischmarkt, Bauernmarkt, \
# Graben and, Morzinplatz near the Temple, according to an Austrian law enforcement \
# source speaking to journalists on Tuesday. Vienna mayor Michael Ludwig said shots \
# appeared to be fired at random, as people dined and drank outside due to the warm \
# weather and virus concerns. Julia Hiermann, who lives in Vienna, was having drinks \
# with a friend when the shooting began. "

# print('----------------------------------------')
# start = time.time()
# ner_map = model.predict(input_seq)
# print(f'total time: {time.time()-start:.2f} seconds')
# print('----------------------------------------')
# print(tabulate(ner_map, headers=['Word', 'NER', 'Confidence']))
# print('----------------------------------------')
#%%
# input_seq = "Dr. Franklin provides us a chance to celebrate Thanksgiving, the Italian way, offering delicious food, wine, music, and dancing.  \
#                 Jackson is in love with Elizabeth.  \
#                 Richard works at VMWare, Georgia. Jackson is from Elizabeh  , New York.     \
#                 Bob and Alice are very close friends. They are also secretely in love! This is a very well kept secret in San Francisco but Tyler knows    \
#                 Dr. Leo Mirani from New York loves to shop at Target"
# input_texts = [input_seq]
# for i in range(7):
#     input_texts.extend(input_texts)

# print()
# print(f'size on disk is {getsizeof(input_texts)} bytes')

# start = time.time()
# ner_maps = []
# for i in range(len(input_texts)):
#     ner_maps += model.predict(input_texts[i], verbose = False)
# print(f'total time: {time.time()-start:.2f} seconds')