import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_validate

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id= '4e8a6305a50f41208b67294e49f5fd05'
client_secret='7b4f7d653d3b4c99b5dec2eda8a76f5e'

class SpotifyMood:
    
    def __init__(self):
            
        self.features_for_mood = ['energy', 'liveness', 'tempo', 'speechiness',
                                            'acousticness', 'instrumentalness', 'danceability', 'duration_ms',
                                            'loudness', 'valence']

        data = pd.read_csv('musicmood.csv')
        hyper_opt = False

        #split into trainval and test
        trainx, testx, trainy, testy = train_test_split(data[self.features_for_mood], data['mood'], test_size = 0.33,
                                                        random_state = 42, stratify=data['mood'])

        self.trainy = trainy

        self.scaler = StandardScaler()
        self.train_scaled = self.scaler.fit_transform(trainx)

        self.nn = MLPClassifier(max_iter = 15000, alpha=1.0, hidden_layer_sizes=8)
        scores = cross_val_score(self.nn, self.train_scaled, trainy, cv=5)
        print ("cv score: " + str(scores.mean()))
        
        client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
        
    def predict(self, playlist_url: str, stress_level: str):
        
        playlist_URI = playlist_url.split("/")[-1].split("?")[0]
        
        data_arr = []

        for track in self.sp.playlist_tracks(playlist_URI)["items"]:
            track_uri = track["track"]["uri"]
            track_data = self.sp.audio_features(track_uri)
            
            data_arr.append([track_data[0][feature] for feature in self.features_for_mood])
            
        dataframe = pd.DataFrame(data_arr, columns=self.features_for_mood)

        results = cross_validate(self.nn, self.train_scaled, self.trainy, return_train_score=True)

        print(results)

        self.nn = MLPClassifier(hidden_layer_sizes=8, max_iter=15000, alpha=1.0)

        self.nn.fit(self.train_scaled, self.trainy)
        test_preds = self.nn.predict(self.scaler.transform(dataframe))
        
        dark_count = np.count_nonzero(test_preds == 'sad')

        # Calculate the percentage of 'dark'
        dark_percentage = dark_count / len(test_preds) * 100

        sadness_prob = dark_count / len(test_preds)

        print("Percentage of sadness based on your favourite song preference :",dark_percentage, "%")

        stress_level = int(stress_level)
        if stress_level<=10 and stress_level>=1 :
            stress_prob = stress_level/10
        else:
            print("Out of range given!")
        print("Probability of stress:", stress_prob)

        depression_prob = sadness_prob * stress_prob

        # print('\n')

        if depression_prob >= 0.75 :
            return {
                'message': 'You are having severe depression! Please contact 1-800-82-0066 or email to info.miasa@gmail.com to recieve help and aid.',
                'suggestion': 'Talk more to people',
                'probability': depression_prob
            }
        elif depression_prob >= 0.35 and depression_prob <0.75 :
            return {
                'message': 'You are having mild depression. If you insist, you can contact 1-800-82-0066 or email to info.miasa@gmail.com to get more information about depression.',
                'suggestion': 'Eat healthy food and find good vibes',
                'probability': depression_prob
            }
        else:
            return {
                'message': 'Your depression type is normal',
                'suggestion': 'Continue having good mental health by doing what you like.',
                'probability': depression_prob
            }