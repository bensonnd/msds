#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify and Genius Data Extract
By: David Wei
Last Modified: 09/23/2021

Current Version: v2.2
- Version v1.0: initial version
- Version v1.1: 
    - adds language checker filter for lyrics to remove non-english songs
    - change incremental song logic from dictionary to dataframe for traceability
    - change export file type from 'csv' to 'pkl'
- Version v2.0: adds logic to check if track already exists
- Version v2.1: appends extract to dedupe (dedupe_songs) 
- Version v2.2: 
    - fixes bug with track_df_final indexs 
    - resets indexes before filtering
    - change logic on removing nonenglish lyrics to filter based on list rather search index
    - removes language filter using TextBlob, use only langdetect

Description: Uses the Spotify and Genius API to make calls to extract random songs, then extracts features for those songs. Once the Spotify song information has been pulled, Genius API is used to extract the lyrics for those songs and filters out non-lyrical and non-english songs then exports unique files to directory. Lastly, all songs prior loaded are loaded in together, deduped and saved as a single (.pkl) file.

Input: no input

Output: 
- song_track_<timestamp>.pkl
- song_album_<timestamp>.pkl
- song_album_cluster_<timestamp>.pkl

- deduped_lyrical_sentiment_load.pkl
- deduped_song_albums.pkl
- load_track_ids.pkl
"""

import sys, os
import requests
from datetime import datetime, date
import re
import random
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
from textblob import TextBlob
from langdetect import detect
from tqdm import tqdm, trange
import pickle
import glob

print('='*100)
startTime = datetime.now()
print(f'Script Starting, hold please...\n')

########################################
#     API init                         #
########################################   

try:
    CLIENT_ID = '07ed80092a9c4446967333aefbefd0b7'
    CLIENT_SECRET = '18dea88e0ebc4339aaba1697a5a5b11a'

    auth_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except: 
    print('Unable to Connect to Spotify, exiting script')
    sys.exist(1)

try:
    # access info
    genius_client_id = 'U54jVZDMgpGq3NP8biCJkkhKNodrWWROURuUqUJ2Te-QzzgbajmHYDcDzyDYMiyi'
    genius_secret_id = '08QbsQmu7vOJC2jtl_m8uCyP_VA2yIxyFcbQAa-pU0iqagr0RGtJiWhfMAyQPc9L4eW-CXRsOR71TtnrYzGWjQ'
    genius_access_token = 'nzn3WwV5iZDeJUENceW4xMgmzBW9OxZvSN5Lya2JnvFUb8YLjC-2MGLdVopcJoI1'

    # init the genius API
    genius = lyricsgenius.Genius(genius_access_token, skip_non_songs=True)
except: 
    print('Unable to Connect to Genius, exiting script')
    sys.exist(1)

########################################
#     code function                    #
########################################   
# Disable stdout printing
class HiddenPrints:
    '''
    Source: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_Random_Search(string):
    '''
    '''
    # gets a random letter from alphabet
    random_letter = random.choice(string.lower())
    # generate a random number between 0 and 2
    random_num = random.randint(0,2)

    # depending on num and search vals, create random wildcard search criteria
    if random_num == 0:
        random_search = random_letter + '%'
    elif random_num == 1:
        random_search = '%' + random_letter + '%'
    elif random_num == 2:
        random_search = random_letter + '%'

    return  random_search

def get_language(text, artist, track):
    try:
        # textblob lang detector
        text_input = TextBlob(text)
        text_language = text_input.detect_language()
    except:
        print(f'Could not get language for: {track} by {artist} using TextBlob language')
        text_language = 'None'
        pass
    
    try:
        #langdetect lang deteector-
        text_language_2 = detect(text)
    except:
        print(f'Could not get language for: {track} by {artist} using LangDetect')
        text_language_2 = 'None'
        pass

    return text_language,text_language_2

def get_audio_features(track_id, feature,target):
    # gets all audio features per track
    try:
        audio_features = sp.audio_features(tracks=[track_id])
        # print(audio_features)
        
        feature = audio_features[0][feature]
    except:
        pass
    # appends the target list
    target.append(feature)

def append_data(iterative, index):
    # print(f'Track#: {num}')
    album_level = iterative['album']

    ######## SONG Level Extraction ########
    # gets track_id
    track_id = str(iterative['id'])

    # check if track_id is already exists
    if track_id in existing_track_ids:
        existing_tracks.append(track_id)
    else:
        track_id_all.append(track_id)
        
        # gets track_name
        track_name = iterative['name']
        # removes the feature artists from track name
        track_name = re.sub("[\(\[].*?[\)\]]", "", track_name).strip()
        #print(track_name)
        track_name_all.append(track_name)

        # get track_duration
        track_duration = iterative['duration_ms']
        track_duration_all.append(track_duration)
        # get track popularity
        track_popularity = iterative['popularity']
        track_popularity_all.append(track_popularity)
        # gets artist_name
        artist_name = album_level['artists'][0]['name']
        artist_name_all.append(artist_name)

        # # gets all audio features per track
        audio_features = ['danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
        features = [danceability_all, energy_all, key_all, mode_all, speechiness_all, acousticness_all, instrumentalness_all, liveness_all, valence_all, tempo_all, time_signature_all]

        for index,(i,j) in enumerate(zip(audio_features,features)):
            audio_feature_val = audio_features[index]
            get_audio_features(track_id, audio_feature_val, j)

        ######## ALBUM Level Extraction ########
        # album ID
        album_id = album_level['id']
        album_id_all.append(album_id)
        # album URI
        album_URI = album_level['uri']
        album_URI_all.append(album_URI)
        # get album name
        album_name = album_level['name']
        album_name_all.append(album_name)

        # # get album language
        # print(album_name)
        # if len(album_name) >= 3:
        #     album_lang = get_language(str(album_name))
        # else:
        #     # assumes english if less than 3 characters long to comply with TextBlob
        #     album_lang = 'en'
        # album_lang_all.append(album_lang)
        
        # get album art
        try:
            album_art = album_level['images'][0]['url']
        except:
            album_art = 'No Album Art Available'
        album_art_all.append(album_art)
        # gets album release_data
        album_release_date = album_level['release_date']
        album_release_date_all.append(album_release_date)
        # total_number of tracks
        album_tot_num_tracks = album_level['total_tracks']
        album_tot_num_tracks_all.append(album_tot_num_tracks)

def remove_df_duplicates(df, info=False):
    dups = df.loc[:, df.columns != 'track_popularity'].duplicated().sum()

    if info == True:
        print(f'Number of duplicates in {df.name} = {dups}')
    elif info == False:
        pass

    df = df.drop_duplicates(subset=df.columns.difference(['track_popularity']),keep='first',inplace=False)
    
    return df
    
def get_song_lyrics(artist, track, info=False):
    if info == True:
        print(f'Track Title: {track}, Artist Name: {artist}').encode("utf-8")
    elif info == False:
        pass

    try:
        # search based on track title and artist name
        lyric_search = genius.search_song(track, artist, get_full_info=False)
        lyric = lyric_search.lyrics
    except:
        lyric = 'None'
        pass

    return lyric

def clean_song_lyrics(lyrics):
    remove_filter_words = re.sub("[\(\[].*?[\)\]]", "", lyrics).strip()
    remove_empty_spaces = str(remove_filter_words).replace('\n\n','').strip()
    remove_watermark = remove_empty_spaces.replace('URLCopyEmbedCopy', '').strip()
    remove_embed_watermark = remove_watermark[:remove_watermark.rfind(' ')]
    return remove_embed_watermark

########################################
#     spotify extract                  #
########################################
spotify_startTime = datetime.now()   

search_type ='track'
characters = 'abcdefghijklmnopqrstuvwxyz'

# extract configurables
num_searches = 1000
search_limit = 50
tot_search = num_searches*search_limit
print(f'Starting Spotify Extract:\nNumber of Searches:{num_searches}\nSearch Limit:{search_limit}\nTotal # of Searches:{tot_search}\n')


### song level ###
track_id_all = []
track_name_all = []
artist_name_all = []
track_duration_all = []
track_popularity_all = []

### audio features ###
danceability_all = []
energy_all = []
key_all = []
mode_all = []
speechiness_all = []
acousticness_all = []
instrumentalness_all = []
liveness_all = []
valence_all = []
tempo_all = []
time_signature_all = []

### album level ###
album_id_all = []
album_URI_all = []
album_name_all = []
# album_lang_all = []
album_art_all = []
album_release_date_all = []
album_tot_num_tracks_all = []

### Existing Track Count
existing_tracks = []

# Loaded Filter Logic'
existing_track_ids_load = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/load_track_ids.pkl', 'rb'))
existing_track_ids = list(existing_track_ids_load)

for search_count in trange(0,num_searches):
    # print(f'search_count = {search_count}')
    
    # gets random search criteria and random offset
    search_criteria = get_Random_Search(characters)
    offset = random.randint(1,999)

    try:
        results = sp.search(q=search_criteria, type=search_type,offset=offset, market='US',limit=search_limit)
        # print(f'Search Criteria: {search_criteria}\nOffset Criteria:{offset}')
    except: 
        print(f'Could not call API')
        pass

    for num,i in enumerate(results['tracks']['items']):
        append_data(i,num)

print(f'# of Tracks Already Extracted: {len(existing_tracks)}')

# track_columns = ['track_id', 'track_name', 'album_id', 'artist_name', 'track_duration' 'track_popularity', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
track_df_final = pd.DataFrame({
    'track_id': track_id_all,
    'track_name': track_name_all,
    'album_id': album_id_all,
    'album_name': album_name_all,
    # 'album_lang': album_lang_all,
    'artist_name': artist_name_all,
    'track_duration': track_duration_all,
    'track_popularity': track_popularity_all,
    'danceability': danceability_all, 
    'energy':energy_all,
    'key':key_all,
    'mode':mode_all,
    'speechiness': speechiness_all,
    'acousticness': acousticness_all,
    'instrumentalness': instrumentalness_all,
    'liveness': liveness_all,
    'valence': valence_all,
    'tempo': tempo_all,
    'time_signature': time_signature_all})

album_df_final = pd.DataFrame({
    'album_id': album_id_all,
    'album_uri': album_URI_all,
    'album_name': album_name_all,
    # 'album_lang': album_lang_all,
    'album_art_url': album_art_all,
    'album_release_date': album_release_date_all,
    'album_num_tracks': album_tot_num_tracks_all})

# assigning names to the incremental dfs
track_df_final.name = 'track_df_final'
album_df_final.name = 'album_df_final'

print(f'\nDuplicates Found:')
track_df_final = remove_df_duplicates(track_df_final, info=True)
album_df_final = remove_df_duplicates(album_df_final, info=True)

# create df for clustering
album_df_cluster = album_df_final[['album_id','album_art_url']]

print(f'\nExtracted Count:')
print(f'Tracks: {len(track_df_final)}, Albums: {len(album_df_final)}, Albums for Clustering: {len(album_df_cluster)}')


spotify_run_time = datetime.now() - spotify_startTime
print(f'\nSpotify Extract Complete, Time:{spotify_run_time}\n{"-"*80}')

########################################
#     genius extract                  #
########################################
print(f'Starting Genius Lyric Extract:\n')
  
genius_startTime = datetime.now()  

artist_tracks = track_df_final[['artist_name', 'track_name']].to_numpy()

song_lyrics = []
lyric_language = []
lyric_language2 = []

# use Genius API to get the lyrics
for artist,track in tqdm(artist_tracks):
    with HiddenPrints():
        # print(f'track_name:{track}')
        song_lyric = get_song_lyrics(artist, track, info=False)

        #clean up the lyrics a bit
        cleaned_lyrics = clean_song_lyrics(song_lyric)

        # gets the lyric language
        if song_lyric == 'None':
            lyric_lang = 'None'
            lyric_lang2 = 'None'
        else:
            lyric_lang,lyric_lang2 = get_language(song_lyric, artist, track)

        # loop to get all song lyrics and the language detected
        song_lyrics.append(cleaned_lyrics)
        lyric_language.append(lyric_lang)
        lyric_language2.append(lyric_lang2)


# append the lyrics back to the dataframe
track_df_final['lyrics'] = song_lyrics

# append lyric langauge to dataframe
track_df_final['lang_1'] = lyric_language
track_df_final['lang_2'] = lyric_language2

# removing non-songs (based on average lyric length of 1000)
track_df_final = track_df_final[track_df_final['lyrics'].str.split().str.len().lt(1000)]

# resets the indexes
track_df_final = track_df_final.reset_index(drop=True, inplace=False)
album_df_final = album_df_final.reset_index(drop=True, inplace=False)

print(f'# of tracks: {len(track_df_final)}, # of albums: {len(album_df_final)}')

# filtering out songs based on lyrics
album_id_filter_list = []
for index,i in enumerate(track_df_final.itertuples()):
    track_name_lang_filter = track_df_final['track_name'][index]
    artist_name_lang_filter = track_df_final['artist_name'][index]
    track_id_lang_filter = track_df_final['track_id'][index]
    album_id_lang_filter = track_df_final['album_id'][index]

    # removing songs that are non-english
    # if track_df_final['lang_1'][index] != 'en' and track_df_final['lang_2'][index] != 'en':
    if track_df_final['lang_2'][index] != 'en':
        print(f'Non-English lyric found: "{track_name_lang_filter}" by {artist_name_lang_filter} was deleted')
        
        # append the album_id related to the non-english song
        album_id_filter_list.append(album_id_lang_filter)
        # remove from track dataframe
        track_df_final.drop(index, inplace=True)
        # track_df_final.drop(track_df_final.index[track_df_final['track_id'] == track_id_lang_filter], inplace = True)
        # track_df_final.drop(index=track_df_final.index[index], inplace=True)

        # # remove from album dataframe
        # album_df_final.drop(index, inplace=True)
        # # album_df_final.drop(album_df_final.index[album_df_final['album_id'] == album_id_lang_filter], inplace = True)

for index,i in enumerate(album_df_final.itertuples()):
    album_id_filter = album_df_final['album_id'][index]
    if album_id_filter in album_id_filter_list:
        # remove from album dataframe
        album_df_final.drop(index, inplace=True)

# remove language filter flags
track_df_final = track_df_final.drop(['lang_1', 'lang_2'],axis=1)


genius_run_time = datetime.now() - genius_startTime
print(f'\nGenius Extract Complete, Time:{genius_run_time}')


########################################
#     export files                     #
########################################  

os_name = os.name

# change to extract dir
if os_name == 'posix':
    os.chdir('//Users//david.wei//Google Drive//Masters//Summer 2021 (NLP, QTW, Cap A)//DS 6120 - Capstone A//Data_Files//Spotify_Genius_Extract')

elif os_name == 'nt':
    os.chdir('G://My Drive//Masters//Summer 2021 (NLP, QTW, Cap A)//DS 6120 - Capstone A//Data_Files//Spotify_Genius_Extract//')

# get current timestamp
extract_timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

# # save filenameas CSV
# track_filename = str('song_track_'+str(extract_timestamp)+'.csv')
# album_filename = str('song_album_'+str(extract_timestamp)+'.csv')
# album_cluster_filename = str('song_album_cluster_'+str(extract_timestamp)+'.csv')

# # saves to CSV file
# track_df_final.to_csv(track_filename, index=False, header=True)
# album_df_final.to_csv(album_filename, index=False, header=True)
# album_df_cluster.to_csv(album_cluster_filename, index=False, header=True)

# save filename as PICKLE
track_filename = str('song_track_'+str(extract_timestamp)+'.pkl')
album_filename = str('song_album_'+str(extract_timestamp)+'.pkl')
album_cluster_filename = str('song_album_cluster_'+str(extract_timestamp)+'.pkl')

# saves to pickle file
track_df_final.to_pickle(track_filename)
album_df_final.to_pickle(album_filename)
album_df_cluster.to_pickle(album_cluster_filename)

# gets log_filename
# log_file_name = str('spotify_extract_log-'+str(extract_timestamp)+'.txt')

script_run_time = datetime.now() - startTime
print('-'*100)
print(f'File Exported:\n{track_filename}, {len(track_df_final)} rows\n{album_filename}, {len(album_df_final)} rows\n{album_cluster_filename}, {len(album_df_cluster)} rows\n')
print(f'Script run time: {script_run_time}\nSpotify_Genius_Extract Complete!')
print('-'*100)
########################################
#     Dedupe Songs                     #
########################################  

################## Loading in Data Files ####################
# for songs
print(f'Loading in Data Files:', end=' ')
file_pattern = 'G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Spotify_Genius_Extract/song_track_*'
files = [files for files in glob.glob(file_pattern)]

df = pd.DataFrame()
for track_file in files:
    f = pickle.load(open(track_file, 'rb'))
    df=df.append(f)
    # print(track_file)
print(f'# of songs loaded: {len(df)}')

# for albums
print(f'Loading in Data Files:', end=' ')
album_file_pattern = 'G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Spotify_Genius_Extract/song_album_*'
album_files = [files for files in glob.glob(album_file_pattern)]

album_df = pd.DataFrame()
for album_file in album_files:
    album_f = pickle.load(open(album_file, 'rb'))
    album_df=album_df.append(album_f)
    # print(track_file)
print(f'# of albums loaded: {len(album_df)}\n')

################## Validate Existing Dedupe File ####################
print(f'Checking current dedupe files')
already_deduped_df = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/deduped_lyrical_sentiment_load.pkl', 'rb'))
print(f'Current # of dedupe songs: {len(already_deduped_df)}')

already_deduped_album_df = pickle.load(open('G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/deduped_song_albums.pkl', 'rb'))
print(f'Current # of dedupe albums: {len(already_deduped_album_df)}')

################## Dedupping Songs ####################
# for some reason, Spotify exports will provide the same track_id with different popularity scores? Perhaps 'track_popularity' is a dynamic value.
print(f'\nRemoving Song Duplicates:')
duplicated_rows = df.loc[:, df.columns != 'track_popularity'].duplicated().sum()
print(f'# of Duplicate Songs: {duplicated_rows}')

df = df.drop_duplicates(subset=df.columns.difference(['track_popularity']),keep='first',inplace=False)
print(f'# of songs after dedupe: {len(df)}')

new_songs_added = len(df) - len(already_deduped_df)
print(f'# of new songs added: {new_songs_added}')


################## Dedupping Albums ####################
print(f'\nRemoving Album Duplicates:')
duplicated_rows = album_df.duplicated().sum()
print(f'# of Duplicate albums: {duplicated_rows}')

album_df = album_df.drop_duplicates(keep='first',inplace=False)
print(f'# of albums after dedupe: {len(album_df)}')

new_albums_added = len(album_df) - len(already_deduped_album_df)
print(f'# of new albums added: {new_albums_added}')


################## Exporting Track_Ids for Spotify Extract ####################
export_file_dir = 'G:/My Drive/Masters/Summer 2021 (NLP, QTW, Cap A)/DS 6120 - Capstone A/Data_Files/Lyrical_Sentiment_Dic_Extract/'
dedupe_df_file_name = 'deduped_lyrical_sentiment_load.pkl'
dedupe_album_df_file_name = 'deduped_song_albums.pkl'
dedupe_trackid_file_name = 'load_track_ids.pkl'

print(f'\nExporting deduped song DF: {dedupe_df_file_name}')
df.to_pickle(str(export_file_dir+dedupe_df_file_name))

print(f'\nExporting deduped album DF: {dedupe_album_df_file_name}')
album_df.to_pickle(str(export_file_dir+dedupe_album_df_file_name))


df_pkl = df['track_id']
print(f'Exporting deduped track_id list: {dedupe_trackid_file_name}')
df_pkl.to_pickle(str(export_file_dir+dedupe_trackid_file_name))


print(f'\n Complete!')
print('='*100)