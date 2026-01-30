from flask import Flask, session, redirect, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np

from spotify_api import get_spotify_oauth, get_spotify_client, get_user_top_tracks
from process_user_data import reading_mood
from music_mood_analyzer import final_mood
from process_user_data import process
from models import model

UPLOAD_FOLDER = r'C:\Users\Maria\PycharmProjects\FlaskProject\upload'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#USER_CSV = os.path.join(UPLOAD_FOLDER, "goodreads_library_export.csv")
USER_EMBEDDINGS_PATH = os.path.join("user_embeddings.npy")
LIBRARY_CSV = os.path.join("books.csv")
LIBRARY_EMBEDDINGS_PATH = os.path.join("book_embeddings.npy")

user_books_df = None #pd.read_csv(USER_CSV, encoding='utf-8-sig') if os.path.exists(USER_CSV) else None
user_embeddings = None #np.load(USER_EMBEDDINGS_PATH) if os.path.exists(USER_EMBEDDINGS_PATH) else None

@app.route('/')
def home():
    genres = sorted(pd.read_csv(LIBRARY_CSV)['main_genre'].dropna().unique())
    moods = ['joy', 'sadness', 'anger', 'fear', 'neutral']
    return render_template("index.html", genres=genres, moods=moods)

@app.route('/spotify')
def spotify_login():
    """The rules to get access token are defined. If the user is logged in (has valid token) than we redirect to the page
    with all the top songs listed, otherwise the user has to log in in through authorize_url."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token()):
        authorize_url = sp_oauth.get_authorize_url()
        return redirect(authorize_url)
    return redirect(url_for('spotify_top_tracks'))

@app.route('/top-tracks')
def spotify_top_tracks():
    """Function to get user top tracks from Spotify. If the user is logged in (has valid token) than we can get a
    Spotify client to make the API call for user's top tracks. It is also calculated the general mood of the user based
    on the songs' audio features."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token()):
        authorize_url = sp_oauth.get_authorize_url()
        return redirect(authorize_url)
    sp = get_spotify_client(sp_oauth)
    top_tracks = get_user_top_tracks(sp)
    session['top_tracks'] = top_tracks
    mood=final_mood()
    session['spotify_mood'] = mood
    print(mood)
    return redirect(url_for('recommend'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    """Function to upload a file with extension csv and save in the chosen folder.
    If the user does not select a file, the browser submits an empty file without a filename."""
    global user_books_df, user_embeddings
    if request.method == 'POST':
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            process(file_path)

            user_books_df = pd.read_csv(file_path, encoding='utf-8-sig')
            user_embeddings = np.load(USER_EMBEDDINGS_PATH)

            #flash("File uploaded successfully! Descriptions added and embeddings computed.")
            return redirect(url_for('recommend'))
        #flash('Invalid file type. Please upload a CSV file.')
        return redirect(request.url)
    return render_template('upload.html')


def get_user_moods():
    """Combine Spotify mood and past reading history mood"""
    moods = []
    spotify_mood = session.get('spotify_mood', 'neutral')
    moods.append(spotify_mood)

    if user_books_df is not None:
        book_mood = reading_mood(user_books_df)
        print("Book mood: " + book_mood)
        moods.append(book_mood)
    if not moods:
        moods = ['neutral']
    return moods

def apply_mood_filter(df, moods):
    """Filter books by one or more mood labels."""
    if not moods:
        return df
    moods = [m for m in moods if m in df.columns]
    if not moods:
        return df
    mask = (df[moods] >= 0.5).any(axis=1)
    return df[mask]

def recommend_books(description='', genre='All', mood=None, top_n=10, search_mode=False):
    library_df = pd.read_csv(LIBRARY_CSV, encoding='utf-8-sig')
    library_embeddings = np.load(LIBRARY_EMBEDDINGS_PATH)
    lib_embeddings = library_embeddings / np.linalg.norm(library_embeddings, axis=1, keepdims=True)
    filtered_df = library_df.copy()

    if isinstance(mood, list):
        moods = mood
    elif mood:
        moods = [mood]
    else:
        moods = []

    print("SEARCH MODE =", search_mode)

    if search_mode:
        if description.strip():
            desc_vec = model.encode([description], normalize_embeddings=True)
            similarity_scores = cosine_similarity(lib_embeddings, desc_vec).flatten()
            filtered_df['similarity_input'] = similarity_scores
            filtered_df = filtered_df.sort_values(by='similarity_input', ascending=False)

        if genre != 'All':
            filtered_df = filtered_df[filtered_df['main_genre'] == genre]

        filtered_df = apply_mood_filter(filtered_df, moods)
        return filtered_df.head(top_n)

    if user_books_df is None or user_embeddings is None:
        spotify_mood = session.get('spotify_mood', None)
        moods_to_use = [spotify_mood] if spotify_mood else []
        filtered_df = apply_mood_filter(filtered_df, moods_to_use)
        filtered_df = filtered_df.sort_values(by='average_rating', ascending=False)
        top = filtered_df.head(top_n)
        print("Top books (Spotify-only):")
        print(top[['book_title', 'author', 'average_rating']])
        return top

    similarity_scores = cosine_similarity(lib_embeddings, user_embeddings).mean(axis=1)
    filtered_df['similarity'] = similarity_scores
    filtered_df = apply_mood_filter(filtered_df, moods)
    filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    return filtered_df.head(top_n)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    description = request.form.get('description', '') if request.method == 'POST' else ''
    genre = request.form.get('genre', 'All') if request.method == 'POST' else 'All'
    mood = request.form.get('mood', None) if request.method == 'POST' else None

    user_moods = get_user_moods()
    user_moods = [m for m in user_moods if m]
    print(f"User moods: {user_moods}")
    if mood:
        if isinstance(mood, list):
            user_moods += mood
        else:
            user_moods.append(mood)
    print("FORM description:", description)
    print("FORM genre:", genre)
    print("FORM mood:", mood)
    print("FINAL user_moods:", user_moods)

    search_mode = (description.strip() != '' or genre != 'All' or mood not in [None, ''])
    recommended_books = recommend_books(description=description, genre=genre, mood=user_moods, top_n=10, search_mode=search_mode)
    books_records = recommended_books.to_dict(orient='records')
    return render_template('recommend.html', books=books_records)

@app.route('/callback')
def callback():
    """Spotify redirects the user here after they log in and approve permissions."""
    sp_oauth = get_spotify_oauth()
    sp_oauth.get_access_token(request.args.get('code'))
    return redirect(url_for('spotify_top_tracks'))

@app.route('/logout')
def logout():
    """This function erases everything from the session and redirects back to the home page."""
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
