# %% [markdown]
# ### DS7337 NLP - HW 5
# ### Neil Benson
# %% [markdown]
# <u>**HW 5:**</u>
#
# 1.	Compile a list of static links (permalinks) to individual user movie reviews from one particular website.
#       This will be your working dataset for this assignment, as well as for assignments 7 and 8, which together will make up your semester project.
#
#       a.	It does not matter if you use a crawler or if you manually collect the links, but you will need at least 100 movie review links.
#           Note that, as of this writing, the robots.txt file of IMDB.com allows the crawling of user reviews.
#
#       b.	Each link should be to a web page that has only one user review of only one movie, e.g., the user review permalinks on the IMDB site.
#
#       c.	Choose reviews of movies that are all in the same genre, e.g., sci-fi, mystery, romance, superhero, etc.
#
#       d.	Make sure your collection includes reviews of several movies in your chosen genre and that it includes a mix of negative and positive reviews.
#
# 2.	Extract noun phrase (NP) chunks from your reviews using the following procedure:
#
#       a.	In Python, use BeautifulSoup to grab the main review text from each link.
#
#       b.	Next run each review text through a tokenizer, and then try to NP-chunk it with a shallow parser.
#
#       c.	You probably will have too many unknown words, owing to proper names of characters, actors, and so
#           on that are not in your working dictionary. Make sure the main names that are relevant to the movies
#           in your collection of reviews are added to the working lexicon, and then run the NP chunker again.
#
# 3.	Output all the chunks in a single list for each review, and submit that output for this assignment.
#       Also submit a brief written summary of what you did (describe your selection of genre, your source of reviews,
#       how many you collected, and by what means).

# %%
# imports
import operator
import random
import string
import pprint
from nltk.tokenize import TweetTokenizer
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.matcher import PhraseMatcher
from requests import get
from bs4 import BeautifulSoup
import warnings

# %%
warnings.filterwarnings("ignore")


# %% [markdown]
## Extract Top 10 Thriller Movies from IMDB

# %%


def get_soup(url, headers):
    request = get(url, headers)
    soup = BeautifulSoup(request.content, "html.parser")
    return soup


def get_movies_list(url, headers):
    # gets the list of movies
    movies_soup = get_soup(url, headers).find(class_="lister-list")

    # gets all individual movies
    movies = movies_soup.find_all(class_="lister-item mode-advanced")

    # limiting to the top 10 movies
    return movies[0:10]


def get_movie_links(base_url, movies):
    return [f"{base_url}{movie.find('a').get('href')}" for movie in movies]


def get_movie_titles(movies):
    return [movie.find("img", alt=True).get("alt") for movie in movies]


def get_movie_cast_characters(movie_links, headers):
    cast_character_soup_list = [
        get_soup(link, headers).find(class_="cast_list") for link in movie_links
    ]

    cast_soup = [
        table.find_all(class_="primary_photo") for table in cast_character_soup_list
    ]

    characters_soup = [
        table.find_all(class_="character") for table in cast_character_soup_list
    ]

    cast_names = [
        cast_member.select('[href^="/name"]')[0].find("img", alt=True).get("alt")
        for movie in cast_soup
        for cast_member in movie
    ]

    character_names = [
        str(character.select('[href^="/title"]')[0].contents[0])
        for movie in characters_soup
        for character in movie
        if len(character.select('[href^="/title"]')) > 0
    ]

    return cast_names, character_names


def get_user_review_rating(review):
    if review.find(class_="ipl-ratings-bar"):
        # extract rating if class exists
        return float(review.find("span", attrs={"class": None}).contents[0])
    # default rating to '-1' if no rating exist
    else:
        return -1


def filter_reviews_list(reviews_list):
    filtered_reviews = [
        [
            review
            for review in movie
            # positive reviews
            if get_user_review_rating(review) > 7
            or (
                # negative reviews
                get_user_review_rating(review) < 4
                and get_user_review_rating(review) > 0
            )
        ]
        for movie in reviews_list
    ]

    return filtered_reviews


def get_reviews_list(movie_links, headers):
    reviews_list = [
        get_soup(f"{link}reviews", headers).find_all(class_="review-container")
        for link in movie_links
    ]

    return filter_reviews_list(reviews_list)


def get_user_review_urls(reviews_list, base_url):
    user_review_urls = [
        [
            f"{base_url}{review.find('a', attrs={'class': 'title'})['href']}"
            for review in movie
        ]
        for movie in reviews_list
    ]

    return user_review_urls


def get_user_review_titles(reviews_list):
    user_review_titles = [
        [
            f"{review.find('a', attrs={'class': 'title'}).contents[0]}"
            for review in movie
        ]
        for movie in reviews_list
    ]

    return user_review_titles


def get_user_review_content(user_review_urls, headers):
    user_review_content_list = [
        [
            get_soup(link, headers).find(class_="text show-more__control").contents[0]
            for link in movie
        ]
        for movie in user_review_urls
    ]

    return user_review_content_list


def get_zipped_reviews(user_review_titles, user_review_content, user_review_urls):

    zipped_reviews = [
        list(zip(movie[0], movie[1], movie[2]))
        for movie in list(
            zip(user_review_titles, user_review_content, user_review_urls)
        )
    ]

    return zipped_reviews


def movies_with_user_reviews(zipped_reviews, movie_names):
    movies_with_user_reviews = {
        "movies": [
            {
                "id": idx,
                "title": movie_names[idx],
                "reviews": [
                    {
                        "id": rvw_idx,
                        "review_title": movie_review[0],
                        "content": movie_review[1],
                        "url": movie_review[2],
                    }
                    for rvw_idx, movie_review in enumerate(movie_reviews)
                ],
            }
            for idx, movie_reviews in enumerate(zipped_reviews)
        ]
    }

    return movies_with_user_reviews


def tokenizer(review, tknzr):
    # tokenizes the review and removes punctuation

    tokens = tknzr.tokenize(review)
    remove_punctuation = [i for i in tokens if i not in list(string.punctuation)]

    return " ".join(remove_punctuation)


def get_np_chunks(review_tokens, nlp):
    doc = nlp(review_tokens)
    return doc.noun_chunks


# %% [markdown]
#### Local browser headers to pass along in the GET requests


# %%
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
}


# %% [markdown]
#### Set the thriller movies URL


# %%
thrillers_url = "https://www.imdb.com/search/title/?genres=thriller"
base_url = "https://www.imdb.com"


# %% [markdown]
#### Get a list of movies


# %%
movies = get_movies_list(thrillers_url, headers)


# %% [markdown]
#### Get each movie's page unique URL


# %%
movie_links = get_movie_links(base_url, movies)


# %% [markdown]
#### Get a list of the movie titles, cast, and characters
# The cast and characters will be used to later update the NER lists


# %%
movie_names = get_movie_titles(movies)


# %% [markdown]
#### Extract all the top 25 review objects from IMDB


# %%
reviews_list = get_reviews_list(movie_links, headers)


# %% [markdown]
#### Extract the urls from filtered positive (>7) and negative reviews (<4)


# %%
user_review_urls = get_user_review_urls(reviews_list, base_url)

print(f"Total number of reviews extracted: {sum(len(i) for i in user_review_urls)}")

# %% [markdown]
#### Extract the review title and content (text) for each of the filtered reviews


# %%
user_review_titles = get_user_review_titles(reviews_list)
user_review_content = get_user_review_content(user_review_urls, headers)


# %% [markdown]
#### Combining it all back together


# %%
zipped_reviews = get_zipped_reviews(
    user_review_titles, user_review_content, user_review_urls
)
movies_with_reviews = movies_with_user_reviews(zipped_reviews, movie_names)


# %% [markdown]
#### Reviewing the data pulled from IMDB


# %%
# movies reviewed
pprint.pprint([movie["title"] for movie in movies_with_reviews["movies"]])
print(f"\n")

# a few movies with their respective reviews
pprint.pprint(
    [
        (movie["title"], movie["reviews"][0:2])
        for movie in movies_with_reviews["movies"][0:3]
    ],
    sort_dicts=False,
)


# %% [markdown]
#### Using nltk to tokenize each review, and spaCy to np chunk the tokens


# %%
nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")
matcher = PhraseMatcher(nlp.vocab)
tknzr = TweetTokenizer()

for movie in movies_with_reviews["movies"]:
    for review in movie["reviews"]:
        # word tokenizingn using nltk
        review["review_tokens"] = tokenizer(
            review["review_title"] + review["content"], tknzr
        )

        # noun phrase chunking using spaCy
        review["review_chunks"] = [
            chunk.text for chunk in get_np_chunks(review["review_tokens"], nlp)
        ]

# %% [markdown]
#### Get cast and character names for NER update
# This builds a list of cast and characters. We will later add cast,
# %%
cast_names, character_names = get_movie_cast_characters(movie_links, headers)


# %% [markdown]
#### Train the new NER bc spaCy uses a statistical method to identify named entities
# By adding new named entities to the vocab/lexicon from the corpus of reviews, we hope to see updated NER

# %% [markdown]
# The following functions
#   * add phrase matches to the corpus
#   * offset phrase match from word to character for each review
#   * check the phrase matcher has overlapped any entities, and resolve if so
#   * format each review using spaCy API, including annotations for named entitities
#   * train the NER on the corpus with the formatted reviews
# %%
def set_entity_ner(entity_list, matcher, nlp):
    for i in entity_list[0]:
        matcher.add(entity_list[1], None, nlp.make_doc(i))
    return


def offsetter(lbl, doc, matchitem):
    # offsets word location in text with corresponding letter(s) location
    subdoc = doc[matchitem[1] : matchitem[2]]

    if matchitem[1] == 0:
        string_first = str(subdoc)
        o_one = matchitem[1]

    else:
        string_first = str(doc[0 : matchitem[1]])

        if string_first[-1] == '"' or string_first[-1] == "'":
            o_one = len(string_first)
        else:
            o_one = len(string_first) + 1

    o_two = o_one + len(str(subdoc))
    return (o_one, o_two, lbl)


def match_overlap(matches):
    # before training the model, sometimes spaCy creates overlapping entitities of the same entity
    # this aims to resolve the overlap by creating a single entitity
    # example: "Noah Jupe" was being tagged as "Noah" and "Noah Jupe"; this returns the latter.

    sorted_matches = sorted(matches, key=operator.itemgetter(1, 2))
    match_ranges = [(match[0], range(match[1], match[2])) for match in sorted_matches]
    indeces_to_remove = set()
    new_matches = []

    for index, elem in enumerate(match_ranges):
        if index < (len(match_ranges) - 1):
            this_elem = set(elem[1])
            next_elem = match_ranges[index + 1][1]
            if intersect := this_elem.intersection(next_elem):
                new_matches.append((elem[0], elem[1][0], next_elem[-1] + 1))
                indeces_to_remove.add(index)
                indeces_to_remove.add(index + 1)

    if not indeces_to_remove:
        return sorted_matches

    sorted_matches_dict = {index: value for index, value in enumerate(sorted_matches)}

    for index in indeces_to_remove:
        sorted_matches_dict.pop(index, None)

    sorted_matches = list(sorted_matches_dict.values())
    sorted_matches = sorted(sorted_matches + new_matches, key=operator.itemgetter(1, 2))
    sorted_matches = match_overlap(sorted_matches)

    return sorted_matches


def setup_entity_training_data(sample_data, matcher, nlp):
    # format the ner training data using spaCy api (Example class)
    ner_train_data = []
    for data in sample_data:
        sample_nlp = nlp.make_doc(data)
        if matches := matcher(sample_nlp):
            matches = [
                (nlp.vocab.strings[match_id], start, end)
                for match_id, start, end in matches
            ]
            matches = match_overlap(matches)
            # (o_one, o_two, lbl)
            entities = [offsetter(x[0], sample_nlp, x) for x in matches]
            ner_train_data.append(Example.from_dict(sample_nlp, {"entities": entities}))

    return ner_train_data


# training the model
def train_ner(train_data, nlp):
    # disable pipeline components that won't be changing
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()

        # training for 30 iterations
        for iteration in range(30):

            # shuffling examples before every iteration
            random.shuffle(train_data)

            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(
                    batch,  # batch of texts
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,
                )

    return
    # source: adapted from https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/


# %% [markdown]
#### Update the NER model with the new labels and named entities
# %%
# some titles to add the lexicon
titles_to_add = [
    "The Handmaid's Snail",
    "The Matrix",
    "Matrix",
    "Dumb and Dumber",
    "Blacklist",
    "The Conjuring 1",
    "The Conjuring 2",
    "The Conjuring 3",
    "The Exorcist" "The Conjuring",
]

# some actors to add the lexicon
actors_to_add = ["Wahlberg", "Walberg", "Millicent", "Marky Mark", "Mark", "Chiwetel"]

# some roles to add the lexicon
roles_to_add = ["June"]

# some directors to add the lexicon
directors_to_adddd = ["Bob Zemeckis", "James Wan", "Wan", "Michael Chaves"]

# set their labels
ner_set_labels = [
    (movie_names + titles_to_add, "TITLE"),
    (cast_names + actors_to_add, "ACTOR"),
    (character_names + roles_to_add, "ROLE"),
    (directors_to_adddd, "DIRECTOR"),
]

# %% [markdown]
#### Add cast, character, director, and movie names with their respective labels to NER vocab/lexicon
# %%
for ner in ner_set_labels:
    set_entity_ner(ner, matcher, nlp)


# %% [markdown]
#### Set up the training data to recognize the new named entities in the corpus
# this formats each review using spaCy's API, and demarks named entities within the corpus to train
# %%
# grabbing review content and title data
sample_data = [
    str(review["review_title"] + review["content"])
    for movie in movies_with_reviews["movies"]
    for review in movie["reviews"]
]

# a few test sentences to add to the corpus
sentences_to_add = [
    "Manifest stars Mark Wahlberg, Elisabeth Moss as June Osborne and directed by Wan.",
    "Dominic Toretto isn't a great character in this story arc.",
]

sample_data = sample_data + sentences_to_add

# format the corpus data using spaCy's API
ner_train_data = setup_entity_training_data(sample_data, matcher, nlp)


# %% [markdown]
#### Train the NER model

# %%
# train cast, character, and movie names NER
train_ner(ner_train_data, nlp)

# %% [markdown]
#### Test the new NERs

# %%
# testing with a cast name
# Mark Wahlberg
test_doc_cast = nlp(
    "Mark Wahlberg was the star in Manifest, which also had June Osborne played by Elisabeth Moss and directed by Wan."
)
print("Entities", [(ent.text, ent.label_) for ent in test_doc_cast.ents])

# testing with a character name
# Dominic Toretto
test_doc_role = nlp("One of the worst characters in the movie was Dominic Toretto.")
print("Entities", [(ent.text, ent.label_) for ent in test_doc_role.ents])
# %% [markdown]

# %% [markdown]
#### Chunking again with the new NER's

# %%
# run chunking again, this time with cast and character names added to the NER
for movie in movies_with_reviews["movies"]:
    for review in movie["reviews"]:
        # noun phrase chunking using spaCy with updated lexicon for cast and characters
        review["review_chunks_new"] = [
            chunk.text for chunk in get_np_chunks(review["review_tokens"], nlp)
        ]

# %% [markdown]
#### Compare new chunks to old chunks without new NER's

# %%
# the chunks from the first go-round
review_chunks = [
    review["review_chunks"]
    for movie in movies_with_reviews["movies"]
    for review in movie["reviews"]
]

review_chunks

# the chunks the second go-round with updated NER
review_chunks_new = [
    review["review_chunks_new"]
    for movie in movies_with_reviews["movies"]
    for review in movie["reviews"]
]

review_chunks_new
# %% [markdown]
#### Check difference in chunking
# %%
difference_in_chunking = [
    item for item in review_chunks if item not in review_chunks_new
]
difference_in_chunking

# %% [markdown]
#       Output all the chunks in a single list for each review, and submit that output for this assignment.
#       Also submit a brief written summary of what you did (describe your selection of genre, your source of reviews,
#       how many you collected, and by what means).
#
# Process:
#   * Using BeautifulSoup
#   * Select genre - this was abitrary
#   * Set genre link
#   * Get list of movies (10) from genre page
#   * Get each movie's respective unique URL
#   * Get a list of reviews from movie page
#   * Get each review's respective unique URL
#   * Filter out reviews > 7 or < 4
#   * Check to see if there are at least 100 reviews
#   * Combine movie title, review title, and review content together
#   * Inspect/review a handful of exmamples
#   * Chunk the reviews into NP using spaCy
#   * Add phrase matches to the corpus
#   * Offset phrase match from word to character for each review
#   * Check the phrase matcher has overlapped any entities, and resolve if so
#   * Format each review using spaCy API, including annotations for named entitities
#   * Train the NER on the corpus with the formatted reviews
#   * Test a couple sentences not in the corpus to see how NER worked
#   * Chunk the reviews again into NP using spaCy after adding NER's to the lexicon
#   * Compare old chunks to new
# Surpising to see that the stock NP chunker was the same before and after adding NER's to the lexicon.
# This speaks to spaCy's chunker, which is statistical in nature.
