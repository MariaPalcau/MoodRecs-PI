import shutil, os, re
import requests, pandas as pd, numpy as np, json
import lyricsgenius
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import pipeline
from tqdm.contrib.concurrent import thread_map

def group(group, size):
    for i in range(0, len(group), size):
        yield group[i:i + size]

def get_audio_features():
    """Function that extracts for each song the valence and energy.
        :return df: Dataframe with the two characteristics for each song."""
    with open("top_tracks.json", "r", encoding="utf-8") as f:
        tracks_data = json.load(f)

    song_ids = [track["id"] for track in tracks_data]
    if not song_ids:
        return pd.DataFrame()

    headers = {'Accept': 'application/json'}
    audio_features = []
    for songs in group(song_ids, 10):
        url = f"https://api.reccobeats.com/v1/audio-features?ids={",".join(songs)}"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            continue
        data_json = response.json()
        if "content" not in data_json:
            continue
        for track in data_json["content"]:
            audio_features.append({
                "id": track["href"].split("/")[-1].split("?")[0],
                "valence": track.get("valence", 0),
                "energy": track.get("energy", 0)
            })
    return pd.DataFrame(audio_features)

def assign_russell_mood(df):
    """To each song is assigned a mood based on Russell's theory.
    :return df: the original Dataframe with the mood for each song."""
    valence = df["valence"]
    energy = df["energy"]
    conditions = [
        (valence >= 0.5) & (energy >= 0.5),
        (valence < 0.5) & (energy >= 0.5),
        (valence >= 0.5) & (energy < 0.5),
        (valence < 0.5) & (energy < 0.5)
    ]
    moods = ['Excited', 'Angry', 'Relaxed', 'Sad']
    df['russell_mood'] = np.select(conditions, moods, default='Unknown')
    return df['russell_mood']

def sanitize_filename(filename):
    """Remove or replace characters not allowed in Windows filenames."""
    return re.sub(r"[^A-Za-z0-9-_.() ]", "_", filename)

def download(entry, lyrics_genius, lyrics_folder):
    """Downloads lyrics for each track and saves them as a text file.
        :param entry: from dict tracks_data.
        :param lyrics_folder: where the lyrics are stored.
        :param lyrics_genius: Genius API client."""
    song = lyrics_genius.search_song(entry["name"], entry["artists"])
    if song:
        title = " - ".join([entry["id"], entry["name"], entry["artists"]])
        safe_title = sanitize_filename(title)
        file_path = os.path.join(lyrics_folder, f"{safe_title}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(song.lyrics)

def get_lyrics():
    """Function to get song lyrics through a genius lyrics API, saved in a folder"""
    load_dotenv()
    client_access_token = os.getenv("CLIENT_ACCESS_TOKEN_GENIUS")
    lyrics_genius = lyricsgenius.Genius(client_access_token)
    lyrics_folder = "songs"
    if os.path.exists(lyrics_folder):
        shutil.rmtree(lyrics_folder)
    os.makedirs(lyrics_folder, exist_ok=True)

    with open("top_tracks.json", "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    #thread_map(lambda entry: download(entry, lyrics_genius, lyrics_folder),tracks_data, max_workers=10)
    for entry in tqdm(tracks_data):
        download(entry, lyrics_genius, lyrics_folder)

def sentiment_lyrics():
    """Function to compute the sentiment for each song based on the lyrics from the songs folder.
    :return df: Dataframe with the sentiment for each song (-1<=score<=1, sentiment in (Negative, Positive, Neutral))."""
    path = 'songs/'
    files = os.listdir(path)
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)
    lyrics_data = []
    for file in files:
        filename = os.path.splitext(file)[0]
        parts = filename.split(" - ", 1)
        song_id = parts[0]
        title_artist = parts[1] if len(parts) > 1 else "Unknown"
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            text = f.read()
        lyrics_data.append({"id": song_id, "title&artist": title_artist, "lyrics": text})
    df = pd.DataFrame(lyrics_data)
    comp_score = []
    sentiment_label = []

    for lyrics in tqdm(df['lyrics']):
        truncated_lyrics = lyrics[:512]
        result = sentiment_model(truncated_lyrics)
        label = result[0]['label']
        score = int(label[0])
        normalized_score = (score - 3) / 2
        comp_score.append(normalized_score)
        sentiment_label.append(label)

    df['comp_score'] = comp_score
    df['sentiment_label'] = sentiment_label
    df['sentiment'] = df['comp_score'].apply(lambda x: 'Positive' if x >= 0.5 else 'Negative' if x <= -0.5 else 'Neutral')
    return df

def lyrics_analysis():
    """Function to get the lyrics and then the sentiment.
    :return df: dataframe with the lyrics and the sentiment"""
    get_lyrics()
    df = sentiment_lyrics()
    return df

def score_to_mood(score):
    """Function to map the sentiment score.
    :return : mood"""
    if score > 0.5:
        return 'joy'
    elif 0.0 <= score <= 0.5:
        return 'neutral'
    elif score < -0.5:
        return 'anger'
    else:
        return 'sadness'

def final_mood():
    """Main function where the dataframe with the mood extracted from audio features and the one from the lyrics
    are combined for each song with different weighting to get the final average mood.
    :return global_score: (mood)"""
    weight_audio = 0.6
    weight_lyrics = 0.4
    with open("top_tracks.json", "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    df_tracks = pd.DataFrame([{
        "id": entry["id"],
        "title": entry["name"],
        "artist": entry["artists"],
        "title&artist": f"{entry['name']} - {entry['artists']}"
    } for entry in tracks_data])
    df_audio_features = get_audio_features()
    df_audio = df_tracks.merge(df_audio_features, on="id", how="left")
    assign_russell_mood(df_audio)
    df_lyrics = lyrics_analysis()
    df_combined=pd.merge(df_audio, df_lyrics[['id', 'comp_score', 'sentiment']], on="id", how="left")
    russell_score = {
        'Excited': 1,
        'Angry': -1,
        'Relaxed': 0.5,
        'Sad': -0.5,
        'Unknown': 0
    }
    df_combined['russell_num'] = df_combined['russell_mood'].map(russell_score)
    df_combined['overall_score'] = df_combined['russell_num'] * weight_audio + df_combined[
        'comp_score'].fillna(0) * weight_lyrics
    print(df_combined)
    global_score = df_combined['overall_score'].mean()
    print(global_score)
    return score_to_mood(global_score)