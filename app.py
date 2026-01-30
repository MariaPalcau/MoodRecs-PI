from flask import Flask, session, redirect, render_template, request, url_for
import os, glob
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np

from spotify_api import get_spotify_oauth, get_spotify_client, get_user_top_tracks
from process_user_data import reading_mood
from music_mood_analyzer import final_mood
from process_user_data import process, model

UPLOAD_FOLDER = os.path.join(os.getcwd(), "upload")
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

USER_EMBEDDINGS_PATH = os.path.join("user_embeddings.npy")
LIBRARY_CSV = os.path.join("books.csv")
LIBRARY_EMBEDDINGS_PATH = os.path.join("book_embeddings.npy")

user_books_df = None
user_embeddings = None

@app.route('/')
def home():
    genres = sorted(pd.read_csv(LIBRARY_CSV)['main_genre'].dropna().unique())
    moods = ['joy', 'sadness', 'anger', 'fear', 'neutral']
    return render_template("index.html", genres=genres, moods=moods)

@app.route('/spotify')
def spotify_login():
    """The rules to get access token are defined. If the user is logged in then we redirect to the next page,
     otherwise the user has to log in through authorize_url."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token()):
        authorize_url = sp_oauth.get_authorize_url()
        return redirect(authorize_url)
    return redirect(url_for('spotify_top_tracks'))

@app.route('/top-tracks')
def spotify_top_tracks():
    """Function to get user top tracks from Spotify. If the user is logged in then we can get a
    Spotify client to make the API call for user's top tracks. It is also calculated the general mood of the user based
    on the songs' audio features."""
    sp_oauth = get_spotify_oauth()
    if not sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token()):
        authorize_url = sp_oauth.get_authorize_url()
        return redirect(authorize_url)
    sp = get_spotify_client(sp_oauth)
    get_user_top_tracks(sp)
    mood=final_mood()
    session['spotify_mood'] = mood
    print(mood)
    return redirect(url_for('recommend'))

def cleanup_upload_folder():
    """Remove all files in upload folder."""
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    """Function to upload a file with extension csv and save in the chosen folder."""
    global user_books_df, user_embeddings
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            cleanup_upload_folder()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            process(file_path)
            user_books_df = pd.read_csv(file_path, encoding='utf-8-sig')
            user_embeddings = np.load(USER_EMBEDDINGS_PATH)
            return redirect(url_for('recommend'))
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
    """Filter books by one or more mood labels that have a score of at least 70."""
    if not moods:
        return df
    moods = [m for m in moods if m in df.columns]
    if not moods:
        return df
    mask = (df[moods] >= 0.7).any(axis=1)
    return df[mask]

def diversification(df, top_n, k):
    """To diversify the recommendations a group of a chosen number of books are selected and only a part displayed."""
    top_k_df = df.head(k)
    return top_k_df.sample(n=top_n, random_state=None)

def apply_mood_boost(df, moods):
    """Boost similarity score using mood columns."""
    boost = 0.2
    for mood in moods:
        df["similarity"] += df[mood] * boost
    return df

def recommend_books(description='', genre='All', mood=None, top_n=12, search_mode=False):
    """Main function for recommending books. If the search_mode is active then the description is embedded and
    searched based on similarity, genre and mood. If the user doesn't upload his reading history but gives access to his
    top tracks the recommendations are based on the mood from them. If the user uploaded his reading history then the
    filtering is done by the similarity score and this score can be boosted using mood. All go through the diversification
    process.
    :param description: Description of the book to get the recommendations for.
    :param genre: Genre of the book to get the recommendations for.
    :param mood: Mood to get the recommendations for.
    :param top_n: Number of books to get recommendations for.
    :param search_mode: Boolean value to decide whether to search for books through the form or not.
    :return filtered_df: DataFrame with the recommendations."""
    library_df = pd.read_csv(LIBRARY_CSV, encoding='utf-8-sig')
    library_embeddings = np.load(LIBRARY_EMBEDDINGS_PATH)
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
            similarity_scores = cosine_similarity(library_embeddings, desc_vec).flatten()
            filtered_df['similarity'] = similarity_scores
            filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
        if genre != 'All':
            filtered_df = filtered_df[filtered_df['main_genre'] == genre]
        filtered_df = apply_mood_filter(filtered_df, moods)
        return filtered_df.head(top_n)
    if user_books_df is None or user_embeddings is None:
        spotify_mood = session.get('spotify_mood', 'neutral')
        moods_to_use = [spotify_mood] if spotify_mood else []
        filtered_df = apply_mood_filter(filtered_df, moods_to_use)
        filtered_df = filtered_df.sort_values(by='average_rating', ascending=False)
        print("Top books (Spotify-only):")
        print(filtered_df[['book_title', 'author', 'average_rating']])
        return filtered_df.head(top_n)
    similarity_scores = cosine_similarity(library_embeddings, user_embeddings).mean(axis=1)
    filtered_df['similarity'] = similarity_scores
    filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    filtered_df = diversification(filtered_df, top_n=top_n, k=30)
    filtered_df = apply_mood_boost(filtered_df, moods)
    return filtered_df.head(top_n)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Function to get recommended books."""
    description = request.form.get('description', '') if request.method == 'POST' else ''
    genre = request.form.get('genre', 'All') if request.method == 'POST' else 'All'
    mood = request.form.get('mood', None) if request.method == 'POST' else None

    user_moods = get_user_moods()
    user_moods = [m for m in user_moods if m]
    if mood:
        if isinstance(mood, list):
            user_moods += mood
        else:
            user_moods.append(mood)
    print(f"User moods: {user_moods}")
    print("FORM description:", description)
    print("FORM genre:", genre)
    print("FORM mood:", mood)
    print("FINAL user_moods:", user_moods)

    search_mode = (description.strip() != '' or genre != 'All' or mood not in [None, ''])
    recommended_books = recommend_books(description=description, genre=genre, mood=user_moods, top_n=12, search_mode=search_mode)
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
    global user_books_df, user_embeddings
    user_books_df = None
    user_embeddings = None
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)