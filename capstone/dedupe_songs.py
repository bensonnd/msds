#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dedupe_songs
Version: v1.2

Description: 
- Step 1: compile and dedupe all extracted spotify files

Output:
- deduped_lyrical_sentiment_load.pkl, load_track_ids.pkl
"""
import pickle
import glob
import pandas as pd

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