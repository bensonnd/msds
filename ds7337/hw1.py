import requests
import nltk
import re
import pprint

nltk.download("punkt")


def get_text_tokens(decoded_text):
    tokens = nltk.word_tokenize(decoded_text)
    text = nltk.Text(tokens)
    return text


def text_getter(target_url):
    """
    Retrieves book text from a url and returns decoded text for nl analysis
    param:
        target_url (str): the url to retrieve the book text from
    return:
        decoded_text: the book text decoded
    """
    response = requests.get(target_url)
    # response.encoding = "utf-8"
    decoded_text = response.text

    return book_title(decoded_text), get_text_tokens(decoded_text)


def lexical_diversity(book_text):
    """
    Calculates the lexical diversity of the book text.
    param:
        book_text: The text of the book to calculate word count and vocaulary size
    return:
        diversity_score: The unique number of words in a book divided by the total words used in the book.

    """
    vocabulary_size = len(set(book_text))
    total_word_count = len(book_text)
    lexical_diversity = vocabulary_size / total_word_count
    return lexical_diversity


def book_title(book_text):
    """
    Extracts the title of a book.
    param:
        book_text: The text of the book to extract the book title from.
    return:
        title: The parsed title of the book.

    """
    search = re.search("Title:(.*)", book_text)
    title = search.group(1).replace("\r", " ").strip()
    return title


urls = [
    "https://www.gutenberg.org/cache/epub/16751/pg16751.txt",
    "https://www.gutenberg.org/cache/epub/14640/pg14640.txt",
    "https://www.gutenberg.org/cache/epub/19923/pg19923.txt",
]

books = [text_getter(url) for url in urls]

lex_diversity = {
    title: {
        "lexical_diversity": f"{lexical_diversity(text)*100:.2f}",
        "vocabulary_size": len(set(text)),
    }
    for title, text in books
}

for key, value in lex_diversity.items():
    print(f"{key}:\n  {value}\n")
