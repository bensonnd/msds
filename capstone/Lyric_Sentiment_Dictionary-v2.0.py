#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lyric_Sentiment_Dictionary
By: David Wei
Last Modified: 09/21/2021

Current Version: v2.0
- Verison v1.0: original version
- Version v1.1: changes unmatched logic to account to normalize unmatched and matched word set
- Version v1.2: adds dedupe process
- Version v2.0: removes the dedupe process from v1.2 and appends it to Spotify_Genius_Extract script

Description: 
- Step 1: load in compiled and deduped song (total) list
- Step 2: cleans songs, fetch features, create index of unmatched words
- Step 3: predict on unmatched words using GradientBoostingRegressor model
- Step 4: compile ANEW + predicted sentiment dictionary

Input: 
- deduped_lyrical_sentiment_load

Output:
- lyric_sentiment_dictionary.npy
"""

import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, date
import matplotlib.pyplot as plt
import glob

from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import reuters, wordnet
from langdetect import detect


from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


print('='*100)
script_startTime = datetime.now()
print(f'Script Starting, hold please...\n')  
# from pycaret.regression import *
# from sklearnex import patch_sklearn
# patch_sklearn()

def save_dictionary(filename, dictionary_name):
    np.save(filename, dictionary_name)

def load_dictionary(filename):
    dic = np.load(filename, allow_pickle=True).item()
    return dic

def tokenize_text(text):
    tweet_tokenizer = TweetTokenizer()
    text_tokenized = tweet_tokenizer.tokenize(text)
    return text_tokenized
    
def remove_stopwords(text):
    # remove stopwords
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = [w for w in text if not w in stop_words]
    return filtered_sentence

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(w) for w in text]
    return lemmatized_text

def count_senses(word):
    return len(wordnet.synsets(word))

def count_syllables(word):
    vowels = 'aeiouy'

    vowel_count = 0
    for i in range(len(word)):
        if word[i] in vowels and (i == 0 or word[i-1] not in vowels):
            vowel_count +=1
    return vowel_count

def get_valence(word):
    sid = sia()
    valence_val = sid.polarity_scores(word)
    return valence_val

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
    
    return dic

def view_matrix(matrix, index_df):
    # convert document-term matrix to array 
    tfidf_array = matrix.toarray()
    # tokenize vectors to get the actual term (movie title) names
    tokens = vectorizer.get_feature_names()

    #### Converts tokens to DF ####
    # doc_names = [f'book_{i+1}' for i, _ in enumerate(matrix)]
    doc_names = [index_df[i] for i, _ in enumerate(tfidf_array)]
    df = pd.DataFrame(data=tfidf_array, index=doc_names, columns=tokens)
    return df,tokens

def get_language(text):
    try:
        #langdetect lang deteector-
        text_language = detect(text)
        # print(text_language)
    except:
        # print(f'Could not get language for: {text} using LangDetect')
        text_language = 'None'
        pass

    return text_language

def onehot_encode(df, col):
    encoded_pos_tag = pd.get_dummies(df[col])

    #joins one-hot encoded to df
    encoded_df = df.join(encoded_pos_tag)

    # removes original column
    del encoded_df[col]

    return encoded_df

def scale_to_2d(val):
    normalized_val = -1 + 2*(val - 0)/(1 - 0)
    return normalized_val

################## Loading in Data Files ####################
print(f'Loading in Deduped Data Files:', end=' ')
df = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/deduped_lyrical_sentiment_load.pkl', 'rb'))
print(f'# of songs loaded: {len(df)}\n')

################## Loading in normalized ANEW Sentiment Dictionary ####################
print(f'\nLoading in Normalized Sentiment Dictionary:', end= ' ')
sentiment_dic = load_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dictionary_normalized.npy')
print(f'# of words loaded: {len(sentiment_dic)}')

#################### Cleaning Lexicon ####################
lyrics = df['lyrics'].to_list()
song_names = df['track_name'].to_list()

print(f'\nCleaning up the Song Lyrics Lexicon:', end=' ')
cleaned_lyrics = []
for song_lyric in tqdm(lyrics):
    # removes punctuation
    song_lyric = ''.join(word for word in song_lyric if word not in  ("?", ".", ";", ":", "!", ","))
    # converts to lowercase
    unique_lyric_words = list(set(song_lyric.replace('\n',' ').lower().split(' ')))
    # join together the words
    lyric_words = ' '.join(unique_lyric_words)
    # tokenize text
    tokenized_text = tokenize_text(lyric_words)
    # remove stopwords
    filtered_text = remove_stopwords(tokenized_text)
    # lemmatize text
    lemmatized_text = lemmatize_text(filtered_text)
    # final cleanup
    cleaned_text = [text for text in lemmatized_text if text not in ['', "'"]]

    cleaned_lyrics.append(cleaned_text)


#################### Getting Word Features ####################
print(f'\nExtracting Song Lyric Features:')
# Getting the Word Synsets (# of senses a word has)
song_synsets = []
song_syllables = []
song_pos_tag = []

# for list of words in each song in list of songs
for song in tqdm(cleaned_lyrics):
    # print(song)
    # print(f'song #: {track_idx}')
    total_word_synset = []
    total_num_syllables = []
    total_pos_tags = []

    ######### getting the POS tag #########
    word_pos_tag = nltk.pos_tag(song)
    # print(word_pos_tag)

    # for every word in list of words for each song ... get POS tag
    for pos_tag in word_pos_tag:
        pos_tag = pos_tag[1]
        # given prior research on sentiments: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.653&rep=rep1&type=pdf
        # only nouns, verbs, adjectives, and adverbs can have emotional meaning
        required_pos_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        if pos_tag in required_pos_tags:
            total_pos_tags.append(pos_tag)  
        elif pos_tag not in required_pos_tags:
            pos_tag = 'Useless'
            total_pos_tags.append(pos_tag)
        
    song_pos_tag.append(total_pos_tags)

    for word in song:
        ######### getting the synset (# of senses) #########
        word_synset = count_senses(word)
        total_word_synset.append(word_synset)

        ######### getting the syllables per word #########
        num_syllables = count_syllables(word)
        total_num_syllables.append(num_syllables)


    song_synsets.append(total_word_synset)
    song_syllables.append(total_num_syllables)

#################### Filtering out Lists ####################
print(f'\nFiltering Feature Data for Relevant POS Tags:')
for song_idx, pos_tag in enumerate(tqdm(song_pos_tag)):
    for tag_idx,tag in enumerate(pos_tag):
        if tag == 'Useless':
            # remove index from song list
            cleaned_lyrics[song_idx].pop(tag_idx)
            # remove index from song_synsets list
            song_synsets[song_idx].pop(tag_idx)
            # remove index from song_syllables list
            song_syllables[song_idx].pop(tag_idx)
            # remove index from song_pos_tag list
            song_pos_tag[song_idx].pop(tag_idx)

#################### Getting Unmatched Words ####################
print(f'\nExtracting Unmatched Words:')
# storing unmatched sentiment words
unmatched_sentiment_words = set()
matched_sentiment_words = set()

for song_idx, song in enumerate(tqdm(cleaned_lyrics)):

    total_matched_sentiment_words = []
    total_unmatched_sentiment_words = []
    
    for word_idx, word in enumerate(song):
        # lookup if word exist in labeled sentiment dictionary, if so do nothing
        if word in sentiment_dic:
            total_matched_sentiment_words.append(word) 
            # pass
        else: 
            total_unmatched_sentiment_words.append(word)

    unmatched_sentiment_words.update(total_unmatched_sentiment_words)
    matched_sentiment_words.update(total_matched_sentiment_words)
print(f'Number of Unmatched Words: {len(unmatched_sentiment_words)}')
print(f'Number of Matched Words: {len(matched_sentiment_words)}')
total_words_matched_perc = len(matched_sentiment_words)/(len(matched_sentiment_words)+len(unmatched_sentiment_words))*100

print(f'% of Total Words Matched: {round(total_words_matched_perc, 3)}')

#################### Prepping Unmatched Words ####################
unmatched_sentiment_words = list(unmatched_sentiment_words)
print(f'\nPrepping Unmatched Words:')
english_unmatched_sentiment_words = []
for word in tqdm(unmatched_sentiment_words):
    # removing korean words
    word_lang = get_language(word)
    if word_lang == 'ko':
        pass
    else: 
        english_unmatched_sentiment_words.append(word)

# print(english_unmatched_sentiment_words)

#################### Feature Tagging Unmatched Words ####################
print(f'\nFeature Tagging Unmatched Words:')
unmatched_dic = {}
for word in tqdm(english_unmatched_sentiment_words):
    unmatched_dic[word] = []

    # getting valence values
    valence_val = get_valence(word)['compound']
    unmatched_dic[word].append(valence_val)

    # set arousal to 0
    arousal_val = 0
    unmatched_dic[word].append(arousal_val)

# get feature values added to unmatched_dic
unmatched_dic = add_dic_features(unmatched_dic)

#################### Getting Difference between Matched and Unmatched Dictionaries ####################
unq_unmatched_POS = set();
for i in unmatched_dic.values():
    unq_unmatched_POS.add(i[4])

unq_matched_POS = set();
for i in sentiment_dic.values():
    unq_matched_POS.add(i[4])

different_POS_unmatched = unq_matched_POS.difference(unq_unmatched_POS)
different_POS_matched = unq_unmatched_POS.difference(unq_matched_POS)
print(f'Difference between Matched & Unmatched POS Tags: {different_POS_unmatched}')
print(f'Difference between Unmatched & Matched POS Tags: {different_POS_matched}')

#################### Creating Dataframe for Modeling ####################
print(f'\nCreating Dataframe for Modeling:')
sentiment_dic_colnames = [val for val in sentiment_dic.keys()]
unmatched_dic_colnames = [val for val in unmatched_dic.keys()]

# creating dataframe for sentiment dictioanry
sentiment_dic_df = pd.DataFrame.from_dict(sentiment_dic, orient='index', columns=['Valence', 'Arousal', 'Senses', 'Syllables', 'POS_Tag', 'Word_Count'])
sentiment_dic_df.insert(0, 'Word', sentiment_dic_colnames)
print(f'Count of Labeled Dictionary: {len(sentiment_dic_df)}')

# creating dataframe for words unmatched by sentiment dictionary
unmatched_dic_df = pd.DataFrame.from_dict(unmatched_dic, orient='index', columns=['Valence', 'Arousal', 'Senses', 'Syllables', 'POS_Tag','Word_Count'])
unmatched_dic_df.insert(0, 'Word', unmatched_dic_colnames)
print(f'Count of Unmatched: {len(unmatched_dic_df)}')

#################### One-Hot Encode the POS_Tag ####################
encoded_sentiment_dic_df = onehot_encode(sentiment_dic_df, 'POS_Tag')
encoded_unmatched_dic_df = onehot_encode(unmatched_dic_df, 'POS_Tag')

####################  Aligning Unmatched and Matched df's ####################
print(f'\nValidating Unmatched vs Matched Column Count:', end= ' ')
# adding different POS tags to unmatched dictioanry, init with 0
for i in different_POS_unmatched:
    encoded_unmatched_dic_df[i] = 0

for i in different_POS_matched:
    encoded_sentiment_dic_df[i] = 0

if len(encoded_sentiment_dic_df.columns) != len(encoded_unmatched_dic_df.columns):
    print('Validation FAILED!, Exiting Code')
    print(f'Sentiment Dic # Cols: {len(encoded_sentiment_dic_df.columns)},Unmatched Dic # Cols: {len(encoded_unmatched_dic_df.columns)}')
    sys.exit()
else:
    print('Validation Passed!')
    print(f'Sentiment Dic # Cols: {len(encoded_sentiment_dic_df.columns)}, Unmatched Dic # Cols: {len(encoded_unmatched_dic_df.columns)}')


#################### Configure Train & Test Split on Sentiment Dictioanry ####################

train,test = train_test_split(encoded_sentiment_dic_df, test_size=.1, shuffle=True, random_state=42)
print(f'train size: {len(train)}\ntest size: {len(test)}')

train.to_pickle('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dic_train.pkl')
test.to_pickle('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dic_test.pkl')

sentiment_dic_train = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dic_train.pkl','rb'))
sentiment_dic_test = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dic_test.pkl','rb'))
print(f'sentiment train size: {len(sentiment_dic_train)},sentiment test size: {len(sentiment_dic_test)}')

model_sentiment_dic_df_train = sentiment_dic_train.drop('Word', axis=1, inplace=False)
model_sentiment_dic_df_test = sentiment_dic_test.drop('Word', axis=1, inplace=False)

cv = KFold(n_splits=10,shuffle=True,random_state=42)

target_column = 'Arousal'
if target_column in model_sentiment_dic_df_train:
    y = model_sentiment_dic_df_train[target_column].values
    del model_sentiment_dic_df_train[target_column]
    X = model_sentiment_dic_df_train.values
if target_column in model_sentiment_dic_df_test:
    y_test = model_sentiment_dic_df_test[target_column].values
    del model_sentiment_dic_df_test[target_column]
    X_test = model_sentiment_dic_df_test.values


if (len(X) != len(y)) or (len(X_test) != len(y_test)):
    print('Issue while Setting up X and Y:', end=' ')
    print(f'Count of X: {len(X)}, Count of y: {len(y)}')
    print(f'Count of X: {len(X_test)}, Count of y: {len(y_test)}')
else:
    print(f'X size: {len(X)}, y size: {len(y)}')
    print(f'X size: {len(X_test)}, y size: {len(y_test)}')
    pass

# for train, test in cv.split(X,y):
#     X_train = X[train] 
#     X_test_body = X[test] 
#     y_train = y[train]
#     y_test  = y[test]

# print(f'X_train: {len(X_train)}\nX_test_body: {len(X_test_body)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}\n')

#################### Building the Model ####################
print(f'\nBuilding the Model...', end= ' ')
# build the model
gbr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.05, loss='ls', max_depth=6,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_decrease=0.0002, min_impurity_split=None,
                          min_samples_leaf=3, min_samples_split=5,
                          min_weight_fraction_leaf=0.0, n_estimators=120,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=42, subsample=0.65, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
# fit the model
gbr.fit(X,y)
print(f'Model Building Complete!')
# evaluate the model
# EvaluateRegressionEstimator(gbr, X, y, cv)

#################### Predicting Unmatched Words ####################
print(f'\nPredicing Unmatched Words:', end=' ')
# predict the model
predict_encoded_unmatched_dic_df = encoded_unmatched_dic_df.drop('Word', axis=1, inplace=False)

target_column = 'Arousal'
if target_column in predict_encoded_unmatched_dic_df:
    y_pred = predict_encoded_unmatched_dic_df[target_column].values
    del predict_encoded_unmatched_dic_df[target_column]
    X_pred = predict_encoded_unmatched_dic_df.values

y_pred = gbr.predict(X_pred)
print(f'# of predicted values: {len(y_pred)}')

#################### Appending Predicted  ####################
print(f'\nAppending Predicted Arousal Values to Unmatched Dictionary')
for idx,(key,value) in enumerate(tqdm(unmatched_dic.items())):
    # print(idx, key, value)
    value[1] = round(y_pred[idx],5)

#################### Creating Collective Sentiment Dictionary  ####################
print(f'\nMerging ANEW and Predicted Sentiment Dictionary:')
song_sentiment_dic = sentiment_dic.copy()
print(f'Size of Sentiment Dictionary (before): {len(sentiment_dic)}')
for idx,(key,value) in enumerate(tqdm(unmatched_dic.items())):
    if key in sentiment_dic:
        pass
    else:
        song_sentiment_dic[key] = value
print(f'Size of Sentiment Dictionary (after): {len(song_sentiment_dic)}')

#################### Exporting Collective Sentiment Dictionary ####################
save_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/lyric_sentiment_dictionary.npy', song_sentiment_dic)

script_run_time = datetime.now() - script_startTime
print(f'\nLyrical Sentiment Dictionary Complete, Run Time:{script_run_time}')
print('='*100)