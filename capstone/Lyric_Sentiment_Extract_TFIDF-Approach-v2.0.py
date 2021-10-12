#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lyric_Sentiment_Extract-TFIDF-Approach
Version: v1.0

Description: Using word vectors and word frequency count in addition to Ensembled Valence to predict overall song Arousal

Steps:
- Step 1: load in compiled and deduped song (total) list
- Step 2: for all words in songs, get valence and arousal values
- Step 3: calculate Overall Ensembled Sentiment Scores
- Step 4: export pkl df for all songs and it's sentiment values

Input: 
- deduped_lyrical_sentiment_load.pkl
- lyric_sentiment_dictionary.npy

Output:
- lyrical_sentiment_df.pkl

By: David Wei
Last Modified: 09/21/2021
"""

import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, date
import matplotlib.pyplot as plt
import math
import glob


from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import reuters, wordnet
from langdetect import detect

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, mean_squared_log_error,r2_score
from sklearn.preprocessing import MinMaxScaler

print('='*100)
script_startTime = datetime.now()
print(f'Script Starting, hold please...\n')  

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

def onehot_encode(df, col, index):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(df[[col]]).toarray(), index=df[index]) 
    encoded_df = df.join(enc_df)

    encoded_df.drop(col, axis=1, inplace=True)
    return encoded_df

def scale_to_2d(val):
    normalized_val = -1 + 2*(val - 0)/(1 - 0)
    return normalized_val

def normalize_sentiment_score(score, alpha=15):
    """
    Source: https://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score/math.sqrt((score*score) + alpha)
    return norm_score

def rescale_pred(val_list):
    scaler = MinMaxScaler(feature_range=(-1,1))
    norm_val = scaler.fit_transform(val_list)
    return norm_val


def rmse(y_actual, y_predicted):
    return np.sqrt(mean_squared_error(y_actual, y_predicted))

def rmsle(y_actual, y_predicted):
    np.sqrt(mean_squared_log_error(y_actual, y_predicted))

def mape(y_actual, y_predicted): 
    mask = y_actual != 0
    return (np.fabs(y_actual - y_predicted)/y_actual)[mask].mean() * 100


#Create scorers for rmse and mape functions
mae_scorer = make_scorer(score_func=mean_absolute_error, greater_is_better=False)
mse_scorer = make_scorer(score_func=mean_squared_error, greater_is_better=False)
rmse_scorer = make_scorer(score_func=rmse, greater_is_better=False)
r2_scorer = make_scorer(score_func=r2_score, greater_is_better=False)
# rmsle_scorer = make_scorer(score_func=rmsle, greater_is_better=False)
# mape_scorer = make_scorer(score_func=mape, greater_is_better=False)

#Make scorer array to pass into cross_validate() function for producing mutiple scores for each cv fold.
errorScoring = {'MAE':  mae_scorer, 
                'MSE': mse_scorer,
                'RMSE': rmse_scorer,
                'R2': r2_scorer,}

    
def EvaluateRegressionEstimator(regEstimator, X, y, cv):
    
    scores = cross_validate(regEstimator, X, y, scoring=errorScoring, cv=cv, return_train_score=True)

    #cross val score sign-flips the outputs of MAE
    # https://github.com/scikit-learn/scikit-learn/issues/2439
    scores['test_MAE'] = scores['test_MAE'] * -1
    # scores['test_MAPE'] = scores['test_MAPE'] * -1
    scores['test_MSE'] = scores['test_MSE'] * -1
    scores['test_RMSE'] = scores['test_RMSE'] * -1
    scores['test_R2'] = scores['test_R2'] * -1

    #print mean MAE for all folds 
    maeAvg = scores['test_MAE'].mean()
    print_str = "The average MAE for all cv folds is: \t\t\t {maeAvg:.5}"
    print(print_str.format(maeAvg=maeAvg))

    #print mean MSE for all folds 
    mseAvg = scores['test_MSE'].mean()
    print_str = "The average MSE for all cv folds is: \t\t\t {mseAvg:.5}"
    print(print_str.format(mseAvg=mseAvg))

    #print mean RMSE for all folds 
    rmseAvg = scores['test_RMSE'].mean()
    print_str = "The average RMSE for all cv folds is: \t\t\t {rmseAvg:.5}"
    print(print_str.format(rmseAvg=rmseAvg))

    #print mean R2 for all folds 
    r2Avg = scores['test_R2'].mean()
    print_str = "The average R2 for all cv folds is: \t\t\t {r2Avg:.5}"
    print(print_str.format(r2Avg=r2Avg))
    # print('*********************************************************')


################## Loading in Data Files ####################
print(f'Loading in Deduped Data Files:', end=' ')
# df = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/lyrical_sentiment_df.pkl', 'rb'))

df = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/deduped_lyrical_sentiment_load.pkl', 'rb'))
# df = df.sample(frac=.01, replace=False, random_state=2)
print(f'# of songs loaded: {len(df)}')

################## Loading in Aggregated Sentiment Dictionary ####################
print(f'Loading in Aggregated Sentiment Dictionary:', end= ' ')
sentiment_dic = load_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/lyric_sentiment_dictionary.npy')
print(f'# of words loaded: {len(sentiment_dic)}')

# ################# Loading in normalized ANEW Sentiment Dictionary ####################
# print(f'\nLoading in Normalized Sentiment Dictionary:', end= ' ')
# sentiment_dic = load_dictionary('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/sentiment_dictionary_normalized.npy')
# print(f'# of words loaded: {len(sentiment_dic)}')

# #################### Cleaning Lexicon ####################
lyrics = df['lyrics'].to_list()
song_names = df['track_name'].to_list()
track_ids = df['track_id'].to_list()
album_ids = df['album_id'].to_list()
artist_name = df['artist_name'].to_list()

album_id = df['album_id'].to_list()
album_name = df['album_name'].to_list()
artist_name = df['artist_name'].to_list()
track_duration = df['track_duration'].to_list()
track_popularity = df['track_popularity'].to_list()
danceability = df['danceability'].to_list()
energy = df['energy'].to_list()
key = df['key'].to_list()
mode = df['mode'].to_list()
speechiness = df['speechiness'].to_list()
acousticness = df['acousticness'].to_list()
instrumentalness = df['instrumentalness'].to_list()
liveness = df['liveness'].to_list()
valence = df['valence'].to_list()
tempo = df['tempo'].to_list()
time_signature = df['time_signature'].to_list()


print(f'\nCleaning up the Song Lyrics Lexicon:')
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


# song_valence_all = []
# for song in cleaned_lyrics:
#     sid = sia()
#     song_string = ' '.join(song)
#     song_valence = sid.polarity_scores(song_string)['compound']
#     song_valence_all.append(song_valence) 



# for song_val in song_valence_all:
#     norm_val = abs(song_val - math.mean(song_val))




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
print(f'% of Total Words Matched: {(len(matched_sentiment_words)/(len(matched_sentiment_words)+len(unmatched_sentiment_words)))*100}')


#################### Filtering Non-english Unmatched Words ####################
unmatched_sentiment_words = list(unmatched_sentiment_words)
print(f'\nFiltering Non-English Unmatched Words:')
english_unmatched_sentiment_words = []
nonenglish_unmatched_words = []

for word in tqdm(unmatched_sentiment_words):
    # removing korean words
    word_lang = get_language(word)
    if word_lang == 'ko':
        nonenglish_unmatched_words.append(word)
    else: 
        english_unmatched_sentiment_words.append(word)


#################### Getting Song Valence & Arousal ####################
print(f'\nFiltering out Unmatched Words, Getting Song Valence & Arousal:')

vader_song_valence = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/get_valence_arousal/vader_song_valence.pkl', 'rb'))
sentiment_dic_song_valence = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/get_valence_arousal/sentiment_dic_song_valence.pkl', 'rb'))
song_arousal = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/get_valence_arousal/song_arousal.pkl', 'rb'))



# vader_song_valence = []
# sentiment_dic_song_valence = []
# song_arousal = []

# for song_idx, song in enumerate(tqdm(cleaned_lyrics)):
#     # for each song iteration, get VALENCE
#     vader_iter_song_valence = []
#     sentiment_dic_iter_song_valence = []
#     # for each song iteration, get AROUSAL
#     iter_song_arousal = []

#     for word_idx,word in enumerate(song):
#         if word in english_unmatched_sentiment_words:      
#             # remove the word from song
#             cleaned_lyrics[song_idx].pop(word_idx)
#         elif word in nonenglish_unmatched_words:
#             # remove the word from song
#             cleaned_lyrics[song_idx].pop(word_idx)
#         else: 
#             # getting VADER valence
#             vader_word_valence_val = get_valence(word)['compound']
#             vader_iter_song_valence.append(vader_word_valence_val)

#             # getting Sentiment Dictionary valence
#             anew_word_valence_val = sentiment_dic[word][0]
#             sentiment_dic_iter_song_valence.append(anew_word_valence_val)

#             # getting Sentiment Dictioanry Arousal
#             word_arousal_val = sentiment_dic[word][1]
#             iter_song_arousal.append(word_arousal_val)

#     vader_song_valence.append(vader_iter_song_valence)
#     sentiment_dic_song_valence.append(sentiment_dic_iter_song_valence)
#     song_arousal.append(iter_song_arousal)

#################### Calculating Overall Ensembled Sentiment Scores ####################
# max_valence_val = max(sentiment_dic.valuex())
# max_arousal_val = 
print(f'\nCalculating Ensembled Valence Score:')
overall_song_valence = []

for song_idx, song in enumerate(tqdm(vader_song_valence)):
    total_overall_valence = []

    for word_idx, valence_val in enumerate(song):
        # first checks VADER, and if neutral..
        if valence_val == 0:
            # then check ANEW, and if ANEW is also neutral
            anew_valence_val = sentiment_dic_song_valence[song_idx][word_idx]
            if anew_valence_val == 0:
                # then the Valence must be neutral
                overall_valence = 0
            else: 
                # and if ANEW isn't neutral, use VADER
                overall_valence = valence_val
        else: 
            # if VADER valence != 0, use VADER
            overall_valence = valence_val
            # then create an ensembled overall valence val
            # overall_valence = valence_val
            # overall_valence = (valence_val + anew_valence_val)/2

        total_overall_valence.append(overall_valence)
    overall_song_valence.append(total_overall_valence)



#################### Calculating Compound Ensembled Sentiment Scores ####################
print(f'Calculating Compound Valence Score:')
compound_song_valence_score = []
overall_song_valence_score = []

for val in tqdm(overall_song_valence):
    sum_val_score = sum(val)
    compound_valence_val = normalize_sentiment_score(sum_val_score)
    compound_song_valence_score.append(compound_valence_val)
    overall_song_valence_score.append(sum_val_score)

print(f'Calculating Compound Arousal Score:')
compound_song_arousal_score = []
overall_song_arousal_score = []

for val in tqdm(song_arousal):
    sum_ars_score = sum(val)
    compound_arousal_val = normalize_sentiment_score(sum_ars_score)
    compound_song_arousal_score.append(compound_arousal_val)
    overall_song_arousal_score.append(sum_ars_score)


# ################## Splitting up the Data ####################
# df_model = df.copy()

# # remove lyrics
# df_model.drop(['lyrics'],axis=1,inplace=True)

# # append cleaned lyrics
# df_model['lyrics'] = cleaned_lyrics


# print(f'\nSplitting up Dataset to Model and Predict')
# df_model,df_model_final = train_test_split(df_model, test_size=.4, shuffle=True, random_state=42)
# print(f'Model Size: {len(df_model)},Predict Size: {len(df_model_final)}')


#################### Creating TFIDF ####################
# model_lyrics = df_model['lyrics']

# using all songs with matched words
cleaned_songs = []
for words in cleaned_lyrics:
    song = ' '.join(words)
    cleaned_songs.append(song)

vectorizer = TfidfVectorizer(strip_accents='ascii')
tfidf_matrix = vectorizer.fit_transform(cleaned_songs)

# convert tfidfmatrix to df
tfidf_df = view_matrix(tfidf_matrix, song_names)[0]

# # add Sentiment Features
# tfidf_df.insert(0, 'Valence', compound_song_valence_score)
# tfidf_df.insert(1, 'Arousal', compound_song_arousal_score)


#################### Topic Modeling with SVD ####################
# remove sentiment features from modeling tfidf
# tfidf_df.drop(['Valence', 'Arousal'],axis=1, inplace=True)

# init SVD LSA
svd_num_components=2500
print(f'\nCreating LSA Model using SVD with {svd_num_components} compoments:')
svd_clf = TruncatedSVD(n_components=2500, n_iter=7, random_state=42)
# fit TFIDF and convert to SVD LSA model
X_lsa = svd_clf.fit_transform(tfidf_df)
print(f'LSA shape: {X_lsa.shape}')

# get variacne ratio of LSA model
var_explained = svd_clf.explained_variance_ratio_.sum()
print(f'SVD LSA Variance Ratio: {var_explained}')

# convert LSA matrix to df
X_lsa_df = pd.DataFrame(X_lsa)

# add Sentiment Features
additional_cols = {'track_id': track_ids,
                    'album_id': album_ids,
                    'valence': compound_song_valence_score,
                    'arousal': compound_song_arousal_score,
                    'song_name': song_names,
                    'artist_name': artist_name,
                    'album_id': album_id,
                    'album_name': album_name,
                    'artist_name': artist_name,
                    'track_duration': track_duration,
                    'track_popularity': track_popularity,
                    'danceability': danceability,
                    'energy': energy,
                    'key': key,
                    'mode': mode,
                    'speechiness': speechiness,
                    'acousticness': acousticness,
                    'instrumentalness': instrumentalness,
                    'liveness': liveness,
                    'spotify_valence': valence,
                    'tempo': tempo,
                    'time_signature': time_signature,
                    }

additional_cols_df = pd.DataFrame(additional_cols)

# X_lsa_df.insert(0, 'Track_ID', track_ids)
# X_lsa_df.insert(1, 'Song_Name', song_names)
# X_lsa_df.insert(2, 'Valence', compound_song_valence_score)
# X_lsa_df.insert(3, 'Arousal', compound_song_arousal_score)

# merge to X_lsa_df
X_lsa_df = additional_cols_df.join(X_lsa_df, how='outer')



# ################## Splitting up the Data ####################
df_model = X_lsa_df.copy()

# # remove lyrics
# df_model.drop(['lyrics'],axis=1,inplace=True)
# # append cleaned lyrics
# df_model['lyrics'] = cleaned_lyrics

print(f'\nSplitting up Dataset to Model and Predict')
df_model,df_predict = train_test_split(df_model, test_size=.6, shuffle=False, random_state=42)
print(f'Model Size: {len(df_model)},Predict Size: {len(df_predict)}')

# remove Arousal from predicted set
df_predict.drop(['arousal'], axis=1, inplace=True)

# set predicted Arousal to 0
df_predict.insert(3, 'arousal', 0)

# pickling file to load later
df_predict.to_pickle('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/df_predict.pkl')

df_model.drop(['track_id', 
                'album_id', 
                'song_name',
                'artist_name',
                'album_id',
                'album_name',
                'artist_name',
                'track_duration',
                'track_popularity',
                'danceability',
                'energy',
                'key',
                'mode',
                'speechiness',
                'acousticness',
                'instrumentalness',
                'liveness',
                'spotify_valence',
                'tempo',
                'time_signature'], axis=1, inplace=True)

df_predict.drop(['track_id', 
                'album_id', 
                'song_name',
                'artist_name',
                'album_id',
                'album_name',
                'artist_name',
                'track_duration',
                'track_popularity',
                'danceability',
                'energy',
                'key',
                'mode',
                'speechiness',
                'acousticness',
                'instrumentalness',
                'liveness',
                'spotify_valence',
                'tempo',
                'time_signature'], axis=1, inplace=True)

# # remove track_id
# df_model.drop(['Track_ID'], axis=1, inplace=True)
# df_predict.drop(['Track_ID'], axis=1, inplace=True)
# # remove song_name
# df_model.drop(['Song_Name'], axis=1, inplace=True)
# df_predict.drop(['Song_Name'], axis=1, inplace=True)


# get column names
# model_col_names = list(tfidf_df.columns)

#################### Configure Train & Test Split on Sentiment Dictioanry ####################
print(f'\nDefining Target and Predictors:')
cv = KFold(n_splits=10,shuffle=True,random_state=42)
print(f'KFold params: {cv}')

target_column = 'arousal'

if target_column in df_model:
    y = df_model[target_column].values
    del df_model[target_column]
    X = df_model.values

if (len(X) != len(y)):
    print('Issue while Setting up X and Y:', end=' ')
    print(f'Count of X: {len(X)}, Count of y: {len(y)}')
else:
    print(f'X size: {len(X)}, y size: {len(y)}')
    pass

#################### Building the Model ####################
print(f'\nBuilding the Ridge Regression Model:')

# build the model
ridge = Ridge(alpha=2.74, copy_X=True, fit_intercept=False, max_iter=None,
      normalize=False, random_state=42, solver='auto', tol=0.001)

# fit the model
ridge.fit(X,y)

# evaluate the model
EvaluateRegressionEstimator(ridge, X, y, cv)

#################### Predicting Song Arousal ####################
print(f'\nPredicting Song Arousal...', end=' ')
# predict the model
# predict_encoded_unmatched_dic_df = final_tfidf.drop('Word', axis=1, inplace=False)

target_column = 'arousal'
if target_column in df_predict:
    y_pred = df_predict[target_column].values
    del df_predict[target_column]
    X_pred = df_predict.values

y_pred = ridge.predict(X_pred)
print(f'# of predicted songs: {len(y_pred)}')

#################### Scaling Predicted Values to 2d plane range ####################
print(f'Rescaling Prediction Results to Standard Range...', end=' ')
y_pred_reshaped = y_pred.reshape(y_pred.shape[0], -1)
y_pred_scaled = rescale_pred(y_pred_reshaped)
y_pred_scaled = y_pred_scaled.flatten()
print(f'Rescaling Complete!')

# #################### Appending Predicted  ####################
print(f'\nAppending Predicted Arousal Values to export dataframe')

df_predict_final = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/df_predict.pkl', 'rb'))
print(f'# of songs loaded: {len(df_predict_final)}')
df_predict_final = df_predict_final.iloc[:,:20]

# remove Arousal from predicted set
df_predict_final.drop(['arousal'], axis=1, inplace=True)

# set predicted Arousal to 0
df_predict_final.insert(3, 'arousal', y_pred_scaled)

# #################### Exporting Collective Sentiment Dictionary ####################
# pickling final file for statistical testing
df_predict_final.to_pickle('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/lyrical_sentiment_df_2.pkl')

script_run_time = datetime.now() - script_startTime
print(f'\nLyrical Sentiment Extract Complete, Run Time:{script_run_time}')