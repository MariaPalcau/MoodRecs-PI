import pandas as pd, requests, re, numpy as np, os
from tqdm import tqdm
from transformers import pipeline
from langdetect import detect
from tqdm.contrib.concurrent import thread_map
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

load_dotenv()
GoogleAPI = os.getenv("GOOGLE_BOOKS_API_KEY")

def clean_isbn (isbn):
    """This function cleans the isbn string and returns it as a string.
    :param isbn: The isbn string to clean.
    :return isbn: Cleaned isbn string."""
    if not isbn:
        return None
    isbn = str(isbn).strip()
    isbn = isbn.strip('="')
    if isbn:
        return isbn
    else:
        return None

def clean_text(text):
    """This function cleans the text string and returns it as a string.
    :param text: The text string to clean.
    :return text: Cleaned text."""
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub("&", "and", text)
    return text.strip().lower()

def get_request(query):
    """This function gets the request from the Google Books API.
    :param query: The query to search for.
    :return description: The requested description from the Google Books API."""
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1&key={GoogleAPI}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    description = None
    for item in data.get("items", []):
        info = item.get("volumeInfo", {})
        if info.get("description"):
            description = info["description"]
            break
    return description

def request_with_isbn(isbn):
    """This function gets the request from the Google Books API using the ISBN.
    :param isbn: The isbn string to search for.
    :return description: The requested description from the Google Books API."""
    query = f"isbn:{isbn}"
    try:
        description = get_request(query)
        return description
    except Exception as e:
        print(f"Error for ISBN {isbn}: {e}")
    return None

def request_with_title_author(title, author):
    """This function gets the request from the Google Books API using the title and the author.
    :param title: The title string to search for.
    :param author: The author string to search for.
    :return description: The requested description from the Google Books API."""
    query = f"{title}+{author}"
    try:
       description = get_request(query)
       return description
    except Exception as e:
        print(f"Error for {title} by {author}: {e}")
    return None

def process_row(row):
    isbn = clean_isbn(row.get('ISBN'))
    description = None
    if isbn:
        description = request_with_isbn(isbn)
    if not description:
        title = clean_text(row.get('Title'))
        author = clean_text(row.get('Author'))
        description = request_with_title_author(title, author)
    return description

def add_description_to_user_books(csv_path):
    """This function adds the description to the user books csv file.
    :param csv_path: The path to the csv file to add."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    mask = df["Exclusive Shelf"].isin(["read"]) & (df["My Rating"].fillna(0) >= 3)
    filtered_df = df.loc[mask]
    descriptions = thread_map(process_row, filtered_df.to_dict("records"), max_workers=10)
    if "Description" not in df.columns:
        df["Description"] = ""
    df["Description"] = df["Description"].astype(str)
    df.loc[mask, "Description"] = descriptions
    df.to_csv(csv_path, index=False)

def calculate_max_emotion_scores(predictions):
    """This function calculates the maximum emotion score per label from a list of predictions.
    :param predictions: List of dicts containing 'label' and 'score'.
    :return emotion_scores: Maximum score for each emotion label."""
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    per_emotion_scores = {label: [] for label in emotion_labels}
    label_score_map = {}
    for prediction in predictions:
        for item in prediction:
            label_score_map[item['label']] = item.get("score", 0.0)
        for label in emotion_labels:
            per_emotion_scores[label].append(label_score_map.get(label, 0.0))
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

def emotion_analysis(csv_path):
    """This function does emotion analysis on book descriptions and updates the CSV.
    It uses two models: one for the english books (it's more accurate) and another for the books in other languages.
    :param: csv_path: Path to the CSV file."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    classifier = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    classifier_support = pipeline(
        "text-classification", model="MilaNLProc/xlm-emo-t", top_k=None)
    book_ids = []
    emotion_scores = {}
    for label in emotion_labels:
        emotion_scores[label] = []
    for i in tqdm(range(len(df))):
        description = df["Description"][i]
        if pd.isnull(description) or not str(description).strip():
            continue
        book_ids.append(df["Book Id"][i])
        sentences = re.split(r"[.!?]+", description)
        lang_detected = detect(description)
        if lang_detected != 'en':
            predictions = classifier_support(sentences)
        else:
            predictions = classifier(sentences)
        max_scores = calculate_max_emotion_scores(predictions)
        for label in emotion_labels:
            emotion_scores[label].append(max_scores[label])
    emotion_df = pd.DataFrame(emotion_scores)
    emotion_df["Book Id"] = book_ids
    df = pd.merge(df, emotion_df, on="Book Id")
    df.to_csv(csv_path, index=False)

def reading_mood(df):
    """Determine the predominant reading mood based on mean emotion scores.
    :param df: DataFrame containing emotion scores for books.
    :return highest_emotion"""
    emotion_labels = ['anger', 'fear', 'joy', 'sadness']
    mean_emotions={}
    for emotion in emotion_labels:
        emotion_values = df[emotion]
        emotion_mean = emotion_values.mean()
        mean_emotions[emotion] = emotion_mean
    highest_emotion = None
    highest_value = 0
    for emotion, value in mean_emotions.items():
        if value > highest_value:
            highest_value = value
            highest_emotion = emotion
    return highest_emotion

def user_books_embedding(csv_path):
    """This function computes and saves the text embeddings for the books read by the user.
    :param csv_path: Path to the CSV file."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['text_for_embedding'] = (
            df['Title'].fillna('') + ' ' +
            df['Author'].fillna('') + ' ' +
            df['Description'].fillna('')
    )
    embeddings = model.encode(df['text_for_embedding'].tolist(), normalize_embeddings=True, show_progress_bar=True)
    np.save('user_embeddings.npy', embeddings)

def process(file_path):
    """Main function to process user books CSV:
        - Add descriptions from Google Books;
        - Perform emotion analysis;
        - Compute embedding.
    :param file_path: Path to the CSV file."""
    add_description_to_user_books(file_path)
    emotion_analysis(file_path)
    user_books_embedding(file_path)
