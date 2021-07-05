# %% [markdown]
# ### DS7337 NLP - HW 6
# ### Neil Benson
# %% [markdown]
# <u>**HW 6:**</u>
#
#        1.	Evaluate text similarity of Amazon book search results by doing the following:
#           a.	Do a book search on Amazon. Manually copy the full book title (including subtitle) of each of the top 24 books listed in the first two pages of search results.
#           b.	In Python, run one of the text-similarity measures covered in this course, e.g., cosine similarity. Compare each of the book titles, pairwise, to every other one.
#           c.	Which two titles are the most similar to each other? Which are the most dissimilar? Where do they rank, among the first 24 results?
#

# %%
# imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings

# %%
warnings.filterwarnings("ignore")


# %% [markdown]
## Question 1
##### Evaluate text similarity of Amazon book search results "Data Mining"
# Do a book search on Amazon. Manually copy the full book title (including subtitle) of each of the top 24 books listed in the first two pages of search results.
cd  # %%
amazon_titles = [
    "Practical Data Science with R",
    "Build a Career in Data Science",
    "Introduction to Statistics: An Intuitive Guide for Analyzing Data and Unlocking Discoveries",
    "Data Mining: Practical Machine Learning Tools and Techniques (Morgan Kaufmann Series in Data Management Systems)",
    "Becoming a Data Head: How to Think, Speak and Understand Data Science, Statistics and Machine Learning",
    "Machine Learning Engineering",
    "Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython",
    "Data Science from Scratch: First Principles with Python",
    "Machine Learning: A Probabilistic Perspective (Adaptive Computation and Machine Learning series)",
    "Introduction to Data Mining (2nd Edition) (What's New in Computer Science)",
    "Introduction to Data Mining and Analytics",
    "Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems)",
    "Data Mining: The Textbook",
    "Learn Data Mining Through Excel: A Step-by-Step Approach for Understanding Machine Learning Methods",
    "Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking",
    "R in Action: Data Analysis and Graphics with R",
    "Data Analytics: Systems Engineering - Cybersecurity - Project Management",
    "Teach Yourself Data Analytics in 30 Days: Learn to use Python and Jupyter Notebooks by exploring fun, real-world data projects",
    "The Hundred-Page Machine Learning Book",
    "Pattern Recognition and Machine Learning (Information Science and Statistics)",
    "The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition (Springer Series in Statistics)",
    "Data Mining for Business Analytics: Concepts, Techniques and Applications in Python",
    "Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management",
    "Data Mining and Machine Learning: Fundamental Concepts and Algorithms",
]

tfidf = TfidfVectorizer().fit_transform(amazon_titles)


# %% [markdown]
# Compare each of the book titles, pairwise, to every other one.

# %%
#  linear_kernel is equivalent to cosine_similarity
#  because the TfidfVectorizer produces normalized vectors.
#  Get the cosine similarities between all documents in the corpus
cosine_similarities = linear_kernel(tfidf, tfidf)

# %% [markdown]
# We can see that each title has a cosine similarity of 1 as it's 100% related to itself
# We can see this at indeces `[0][0]`, `[1][1]`, `[2][2]`, `[3][3]`, and so on
#
# Which two titles are the most similar to each other? Which are the most dissimilar? Where do they rank, among the first 24 results?
# %%
# flatten to get the indices with the highest similarities
cosine_similarities_flatten = linear_kernel(tfidf, tfidf).flatten()
print(cosine_similarities)
# %%
# find the top 6 related documents, use argsort and some negative array slicing
# -25:-26 range becaus we want to ignore first 24 which are just the documents cosine similarities
#  to themselves after sorting

# top two related docs
related_docs_indices = cosine_similarities_flatten.argsort()[-25:-26:-1]
related_docs_indices
# %%
# the cosine similarities of the top 2
cosine_similarities_flatten[related_docs_indices]
# %%
# mapping the flattened indeces back to `amazon_titles`
amazon_titles_indeces_related = [
    (idx // len(amazon_titles), idx % len(amazon_titles))
    for idx in related_docs_indices
]
# the amazon titles indeces for the top 2 most similar titles
amazon_titles_indeces_related

# %%
# top two unrelated docs
unrelated_docs_indices = cosine_similarities_flatten.argsort()[0:1]
unrelated_docs_indices

# %%
# the cosine similarities of the top 2
cosine_similarities_flatten[unrelated_docs_indices]

# %%
# mapping the flattened indeces back to `amazon_titles`
amazon_titles_indeces_unrelated = [
    (idx // len(amazon_titles), idx % len(amazon_titles))
    for idx in unrelated_docs_indices
]
# the amazon titles indeces for the top 2 most dissimilar titles
amazon_titles_indeces_unrelated

# %% [markdown]
# If we explore further, we can see that the cosine similarity pairwise matrix is simetrical with indeces of high cosine similarity in pairs
# such as `(11,3)` and `(3,11)` or `(5,8)` and `(8,5)` and so on


# %%
# reviewing the two most similar titles
for index, element in enumerate(amazon_titles_indeces_related):
    if index % 2 == 0:
        print(
            f"Two most similar docs:\n{amazon_titles[element[0]]}\n{amazon_titles[element[1]]}\n\n"
        )

print(
    f"The two most similar book titles from Amazon search results were ranked {amazon_titles_indeces_related[0][0]} and {amazon_titles_indeces_related[0][1]}."
)
# %%
# reviewing the two most dissimilar titles
for index, element in enumerate(amazon_titles_indeces_unrelated):
    if index % 2 == 0:
        print(
            f"Two most similar docs:\n{amazon_titles[element[0]]}\n{amazon_titles[element[1]]}\n\n"
        )

print(
    f"The two most dissimilar book titles from Amazon search results were ranked {amazon_titles_indeces_unrelated[0][0]} and {amazon_titles_indeces_unrelated[0][1]}."
)

# %% [markdown]
## Question 2
#       2.	Now evaluate using a major search engine.
#           a.	Enter one of the book titles from question 1a into Google, Bing, or Yahoo!. Copy the capsule of the first organic result and the 20th organic result. Take web results only (i.e., not video results), and skip sponsored results.
#           b.	Run the same text similarity calculation that you used for question 1b on each of these capsules in comparison to the original query (book title).
#           c.	Which one has the highest similarity measure?

# %% [markdown]
#### web result 1
# %%
web_results_1 = [
    "Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems)",  # original title
    """Data Mining: Concepts and Techniques (The ... - Amazon
        https://www.amazon.com › Data-Mining-Concepts-Tec...
        Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems) [Han, Jiawei, Kamber, Micheline, Pei, Jian] on ...
    """,
]


tfidf_web_1 = TfidfVectorizer().fit_transform(web_results_1)
cosine_similarities_web_1 = linear_kernel(tfidf_web_1, tfidf_web_1)
# %% [markdown]
# Cosine similarity, original query and 1st result
# %%
cosine_similarities_web_1
# %% [markdown]
#### web result 20
# %%
web_results_20 = [
    "Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems)",  # original title
    """Concepts and Techniques (The Morgan Kaufmann Series in ...
        http://www.gate2biotech.com › data-mining-concepts-a...
        Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems). 
        Authors: Jiawei Han, Micheline Kamber Publishing: ...""",
]


tfidf_web_20 = TfidfVectorizer().fit_transform(web_results_20)
cosine_similarities_web_20 = linear_kernel(tfidf_web_20, tfidf_web_20)
# %% [markdown]
# Cosine similarity, original query and 20th result
# %%
cosine_similarities_web_20
# %% [markdown]
# Cosine similarity comparison between 1st and 20th result
cosine_similarities_web_20[0][1] > cosine_similarities_web_1[0][1]
# %% [markdown]
# The 20th result has a higher similarity to the original query than the first result!
