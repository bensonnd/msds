# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### DS7337 NLP - HW 3
# #### Neil Benson
# %% [markdown]
# # Homework 3
#
# <u>**HW 3:**</u>
#
# [book link](http://www.nltk.org/book/)
#
# 1. Compare your given name with your nickname (if you don’t have a nickname, invent one for this assignment) by answering the following questions:
#
#     - What is the edit distance between your nickname and your given name?
#     - What is the percentage string match between your nickname and your given name?

# Show your work for both calculations.
#
# 2. Find a friend (or family member or classmate) who you know has read a certain book. Without your friend knowing, copy the first two sentences of that book. Now rewrite the words from those sentences, excluding stop words. Now tell your friend to guess which book the words are from by reading them just that list of words. Did you friend correctly guess the book on the first try? What did he or she guess? Explain why you think you friend either was or was not able to guess the book from hearing the list of words.
#
# 3. Run one of the stemmers available in Python. Run the same two sentences from question 2 above through the stemmer and show the results. How many of the outputted stems are valid morphological roots of the corresponding words? Express this answer as a percentage.
#
#

# %%
import nltk
from nltk.stem import PorterStemmer
import requests
import re
from nltk.corpus import stopwords
import Levenshtein as lev
from _cleaning_options.cleaner import simple_cleaner
import re

# %%
nltk.download("stopwords")

# %% [markdown]
# ## Q1, Pt 1 - Edit Distance of Name and Nickname
# %% [markdown]
# Edit distance is the number of edits between two strings that are needed in order for them to be equal. This can include additions, subsitutions, and deletions.
#
# Source: https://www.datasciencelearner.com/nltk-edit_distance-implement-python/

# %%
name = "Neil Benson"
nickname = "Neilybob"


distance = nltk.edit_distance(nickname, name, transpositions=False)
print(f"The edit distance between my name and nickname is {distance}.")

# %% [markdown]
# ## Q1, Pt 2 - Percentage String Match of Name and Nickname
# %% [markdown]
# Using the Levenshtein Distance to measure how far apart two sequences of words are; the minimum amount of these operations that need to be done to name in order to turn it into nickname, correspond to the Levenshtein distance between those two strings.
#
# ![title](https://miro.medium.com/max/875/0*kWblkNhdDJ7XWthC.jpg)
#
# Where i and j are indexes to the last character of the substring we’ll be comparing. The second term in the last expression is equal to 1 if those characters are different, and 0 if they’re the same.
#
# Source: https://towardsdatascience.com/fuzzywuzzy-how-to-measure-string-distance-on-python-4e8852d7c18f
#
#
#
# Thankfully there are packages for this!
#
# %%
ratio = lev.ratio(name.lower(), nickname.lower())
print(f"{ratio*100:.2f}% string match between name and nickname.")

# %% [markdown]
# ## Q2 - Exclude stopwords to guess
# %%
# import text
# The Great Gatsby
book_text_url = "https://www.gutenberg.org/files/64317/64317-0.txt"


# get the text as raw string format
def text_getter(target_url):
    """
    Retrieves book text from a url and returns decoded text for nl analysis
    param:
        target_url (str): the url to retrieve the book text from
    return:
        decoded_text: the book text decoded
    """
    response = requests.get(target_url)
    decoded_text = response.text

    return decoded_text


# replace several substrings at once
def replace_multiple(mainString, toBeReplaces, newString):
    """
    Replace a set of multiple sub strings with a new string in main string.
    """
    # Iterate over the strings to be replaced
    for elem in toBeReplaces:
        # Check if string is in the main string
        if elem in mainString:
            # Replace the string
            mainString = mainString.replace(elem, newString)

    return mainString


# retrieving raw text from book
raw_text = text_getter(book_text_url)

text_no_headers = simple_cleaner(raw_text)

# strip out title and table of contents
regex = r"(^.*\n\s+I\n\n)(.*)"
book_text = re.search(regex, text_no_headers)

# first two sentences - regex not working as expected
first_two_sent = "\n\nIn my younger and more vulnerable years my father gave me some advice\nthat I\x80\x99ve been turning over in my mind ever since.\n\n\x80\x9cWhenever you feel like criticizing anyone,\x80\x9d he told me, \x80\x9cjust\nremember that all the people in this world haven\x80\x99t had the advantages\nthat you\x80\x99ve had.\x80\x9d\n\n"

# cleaning up first 2 sentences
text = " ".join(replace_multiple(first_two_sent.lower(), ["\n", "\r",], " ").split())

# sentence tokenizer
sentence_tokens = nltk.sent_tokenize(text)

# removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")
for i, sentence in enumerate(sentence_tokens):
    sentence_tokens[i] = tokenizer.tokenize(sentence)

# %%
stop_words = set(stopwords.words("english"))

# # filtering out stopwords from text
stop_words_rem = []
for sentence in sentence_tokens:
    new_sentence = []
    for word in sentence:
        if word not in stop_words:
            new_sentence.append(word)
    stop_words_rem.append(new_sentence)

# reviewing the sentences after removing the stop words
for sentence in stop_words_rem:
    print(" ".join(sentence))
# %% [markdown]
# They were not able to guess the book from which these sentences originated. First, when listening to the sentence
# without stop words, they couldn't grasp the concept of putting together a full sentence. They sort of followed along
# but it was difficult.
#
# Second, when I read the sentences in, full they still didn't recognize the book. I confirmed, and it rang a bell,
# but these first two lines were not the most memorable from the book.
#
# Stop words are important. But also, I can see how they can really affect the outcome of NLP efforts.

# %% [markdown]
# ## Q3 - Percentage Match of text using Stemming

# %%
ps = PorterStemmer()


def stemmer(text):
    stems = [ps.stem(word) for word in text]
    return " ".join(stems)


sentence_tokens_txt = [" ".join(sentence) for sentence in sentence_tokens]
stem_sentences_txt = [stemmer(sentence) for sentence in stop_words_rem]

zip_stem_orig_sent = zip(sentence_tokens_txt, stem_sentences_txt)

for sentence in zip(zip_stem_orig_sent):
    ratio_orig = lev.ratio(sentence[0][0], sentence[0][1])
    print(
        f"Stemmed sentence:\n\n  {sentence[0][1]}\n\n  {ratio_orig*100:.2f}% of stem words are morphological roots when calculating Levenshtein Distance ratio.\n"
    )

# %%
