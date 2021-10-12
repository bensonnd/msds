#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalizing ANEW Sentiment Dictionary
Description: converts 0:1 to -1:1 scale for Valence/Arousal metrics
Input: 'NRC-VAD-Lexicon.txt'
Output: 'creates sentiment_dictionary_normalized.npy' file
By: David Wei
Last Modified: 09/21/2021
"""

import numpy as np
import pandas as pd

from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.corpus import wordnet

from sklearn.preprocessing import MinMaxScaler

def count_senses(word):
    return len(wordnet.synsets(word))

def count_syllables(word):
    vowels = 'aeiouy'

    vowel_count = 0
    for i in range(len(word)):
        if word[i] in vowels and (i == 0 or word[i-1] not in vowels):
            vowel_count +=1
    return vowel_count

def add_dic_features(dic):
    for key in dic.keys():
        # gets sysnet
        dic_word_synset = count_senses(key)
        # gets syllables
        dic_num_syllables = count_syllables(key)
        # gets pos_tag
        dic_word_pos_tag = nltk.pos_tag([key])
        # get word count
        dic_word_count = len(key)

        # append dictionary with new Features
        dic[key].append(dic_word_synset)
        dic[key].append(dic_num_syllables)
        dic[key].append(dic_word_pos_tag[0][1])
        dic[key].append(dic_word_count)

def save_dictionary(filename, dictionary_name):
    np.save(filename, dictionary_name)

def load_dictionary(filename):
    dic = np.load(filename, allow_pickle=True).item()
    return dic

def scale_to_2d(val_list):
    normalized_val = -1 + 2*(val_list - 0)/(1 - 0)
    # scaler = MinMaxScaler(feature_range=(-1,1))
    # normalized_val = scaler.fit_transform(val_list)
    return normalized_val



# ################## Add Features to ANEW dictioanry ####################

# read in NRC-VAD Labeled Dic
labeled_sentiment = 'G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/NRC-VAD-Lexicon.txt'

# convert NRC_VAD to df
sentiment_load_df = pd.read_csv(labeled_sentiment, sep='\t')

# remove dominance attributes
# sentiment_load_df.drop('Dominance', axis=1, inplace=True)

# convert to dictioanry
sentiment_load_dic= dict([(i,[a,b,c]) for i, a,b,c in zip(sentiment_load_df.Word, sentiment_load_df.Valence, sentiment_load_df.Arousal, sentiment_load_df.Dominance)])

# cleaning up sentiment dictionary
for key in list(sentiment_load_dic):
    if type(key) != str:
        del sentiment_load_dic[key]


# gets word features
add_dic_features(sentiment_load_dic)


# save ANEW dictioanry with custom word features added
# save_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dictionary.npy', sentiment_load_dic_final)



# ################## Normalizing ANEW dictionary ####################
print(f'\nNormalizing ANEW to 2d scale :')
# normalized_sentiment_dic = load_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dictionary.npy')
normalized_sentiment_dic = sentiment_load_dic.copy()

for key,value in normalized_sentiment_dic.items():
    valence = value[0]
    arousal = value[1]
    dominance = value[2]
 
    normalized_valence = round(scale_to_2d(valence), 5)
    normalized_arousal = round(scale_to_2d(arousal), 5)
    normalized_dominance = round(scale_to_2d(dominance), 5)

    normalized_sentiment_dic[key][0] = normalized_valence
    normalized_sentiment_dic[key][1] = normalized_arousal
    normalized_sentiment_dic[key][2] = normalized_dominance



save_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dictionary_normalized_withDominance.npy', normalized_sentiment_dic)


######## TESTING Normalized Distributions ########
# arousal_vals = []
# valence_vals = []
# for key,value in normalized_sentiment_dic.items():
#     valence = value[0]
#     arousal = value[1]
#     valence_vals.append(valence)
#     arousal_vals.append(arousal)

# arousal_vals_arr = np.array(arousal_vals)
# arousal_vals_arr = arousal_vals_arr.reshape(arousal_vals_arr.shape[0], -1)
# arousal_val_norm = scale_to_2d(arousal_vals_arr)
# arousal_val_norm = arousal_val_norm.flatten()

# valance_vals_arr = np.array(valence_vals)
# valence_vals_arr = valance_vals_arr.reshape(valance_vals_arr.shape[0], -1)
# valence_val_norm = scale_to_2d(valence_vals_arr)
# valence_val_norm = valence_val_norm.flatten()

# fig,ax = plt.subplots()
# ax.hist(sorted(arousal_val_norm), color='lightblue', alpha=0.5, label='ANEW')
# ax.set(title='ANEW Arousal Distribution', xlabel='valence', ylabel='Count of Words')
# plt.legend(loc="upper left")
# plt.show()
