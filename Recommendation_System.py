import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# Load the datasets
books_df = pd.read_csv("G:\CodSoft\Recommendation_System\Books.csv")
movies_df = pd.read_csv("G:\CodSoft\Recommendation_System\Movies.csv")

# Fill NaNs
books_df['authors'] = books_df['authors'].fillna('')
books_df['title'] = books_df['title'].fillna('')
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['title'] = movies_df['title'].fillna('')

# Combine 'title' and 'authors' into a single string for better recommendations in books
books_df['combined_features'] = books_df['title'] + ' ' + books_df['authors']

# Create a TF-IDF Vectorizer to calculate similarity
tfidf_books = TfidfVectorizer(stop_words='english')
tfidf_matrix_books = tfidf_books.fit_transform(books_df['combined_features'])
tfidf_movies = TfidfVectorizer(stop_words='english')
tfidf_matrix_movies = tfidf_movies.fit_transform(movies_df['genres'])

# Calculate the cosine similarity
cosine_sim_books = linear_kernel(tfidf_matrix_books, tfidf_matrix_books)
cosine_sim_movies = linear_kernel(tfidf_matrix_movies, tfidf_matrix_movies)

# Function to get book recommendations
def get_book_recommendations(title):
    idx = books_df[books_df['title'].str.contains(title, case=False, regex=False)].index[0]
    sim_scores = list(enumerate(cosine_sim_books[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar books
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices][['title', 'authors']]

# Function to get movie recommendations
def get_movie_recommendations(genre):
    idx = movies_df[movies_df['genres'].str.contains(genre, case=False, regex=False)].index[0]
    sim_scores = list(enumerate(cosine_sim_movies[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]

# GUI Setup using Tkinter
def welcome_page():
    for widget in root.winfo_children():
        widget.destroy()
    logo_image = Image.open("system_logo.png")
    logo_image = logo_image.resize((180, 160), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = Label(root, image=logo_photo)
    logo_label.image = logo_photo  # Keep a reference
    logo_label.pack(pady=7)
    motto_label = Label(root, text="Find Your Perfect Match", font=("Helvetica", 12, "italic"))
    motto_label.pack(pady=10)
    label = Label(root, text='What are you looking for?', font=('Arial', 14))
    label.pack(pady=20)
    book_button = Button(root, text="Books", command=book_recommendation_page,font=('Arial', 12), width=20, height=2, bg='#818a89', fg='black')
    book_button.pack(pady=5)
    movie_button = Button(root, text="Movies", command=movie_recommendation_page, font=('Arial', 12), width=20, height=2, bg='#333938', fg='white')
    movie_button.pack(pady=5)


def book_recommendation_page():
    for widget in root.winfo_children():
        widget.destroy()
    # Display logo
    logo_image = Image.open("G:/CodSoft/Recommendation_System/Book_Logo.png")
    logo_image = logo_image.resize((160, 150), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = Label(root, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack(pady=10)
    label = Label(root, text='Enter a Book Title', font=('Arial', 12))
    label.pack(pady=10)
    entry = Entry(root, width=50)
    entry.pack(pady=5)
    result_var = StringVar()
    result_label = Label(root, textvariable=result_var, justify=LEFT)
    result_label.pack(pady=10)
    def recommend_books():
        title = entry.get()
        recommendations = get_book_recommendations(title)
        formatted_recommendations = "\n".join([f"• {row['title']} by {row['authors']}" for _, row in recommendations.iterrows()])
        result_var.set("Top 5 Recommendations:\n" + formatted_recommendations)
    button = Button(root, text='Get Recommendations', command=recommend_books, font=('Arial', 12), width=20, height=2, bg='#818a89', fg='black')
    button.pack(pady=20)
    back_button = Button(root, text="Back", command=welcome_page, width=20, height=2, bg='#333938', fg='white')
    back_button.pack(pady=10)

def movie_recommendation_page():
    for widget in root.winfo_children():
        widget.destroy()

    # Display logo
    logo_image = Image.open("G:/CodSoft/Recommendation_System/logo-movie.png")
    logo_image = logo_image.resize((130, 140), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = Label(root, image=logo_photo)
    logo_label.image = logo_photo
    logo_label.pack(pady=10)
    label = Label(root, text='Enter a Movie Genre', font=('Arial', 12))
    label.pack(pady=10)
    entry = Entry(root, width=50)
    entry.pack(pady=5)
    result_var = StringVar()
    result_label = Label(root, textvariable=result_var, justify=LEFT)
    result_label.pack(pady=10)
    def recommend_movies():
        genre = entry.get()
        recommendations = get_movie_recommendations(genre)
        formatted_recommendations = "\n".join([f"• {row['title']} ({row['genres']})" for _, row in recommendations.iterrows()])
        result_var.set("Top 5 Recommendations:\n" + formatted_recommendations)

    button = Button(root, text='Get Recommendations', command=recommend_movies, font=('Arial', 12), width=20, height=2, bg='#818a89', fg='black')
    button.pack(pady=20)
    back_button = Button(root, text="Back", command=welcome_page, width=20, height=2, bg='#333938', fg='white')
    back_button.pack(pady=10)
    
root = Tk()
root.title('Recommendation System')
root.geometry("500x600")
welcome_page()
root.mainloop()
