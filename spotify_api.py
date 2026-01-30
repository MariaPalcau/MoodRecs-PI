from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
from flask import session
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_spotify_oauth():
    """To request the access token to user's data.
    :return sp_oauth: SpotifyOauth object"""
    cache_handler = FlaskSessionCacheHandler(session)
    sp_oauth = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        scope="user-top-read",
        cache_handler=cache_handler,
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        show_dialog=True,
    )
    return sp_oauth

def get_spotify_client(sp_oauth):
    """Once the user has successfully logged in, a Spotify client is created to call the Spotify API.
    :param sp_oauth: SpotifyOauth object
    :return: Spotify client to make API calls"""
    return Spotify(auth_manager=sp_oauth)

def get_user_top_tracks(sp, limit=50):
    """Function to get user top tracks.
    :param sp: Spotify client
    :param limit: Number of tracks to return"""
    tracks_data = []
    top_tracks = sp.current_user_top_tracks(limit=limit, time_range="short_term")["items"]
    for tracks in top_tracks:
        tracks_data.append({
            "id": tracks["id"],
            "name": tracks["name"],
            "artists": tracks["artists"][0]["name"],
        })
        with open("top_tracks.json", "w", encoding="utf-8") as f:
            json.dump(tracks_data, f, ensure_ascii=False, indent=2)
    return None
