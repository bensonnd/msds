# %% [markdown]
# ### DS7337 NLP - HW 4
# #### Neil Benson
# %% [markdown]
# # Homework4
#
# <u>**HW 4:**</u>
#
# [book link](http://www.nltk.org/book/)
#
# 1.	Run one of the part-of-speech (POS) taggers available in Python.
#     - a. Find the longest sentence you can, longer than 10 words, that the POS tagger tags correctly. Show the input and output.
#     - b. Find the shortest sentence you can, shorter than 10 words, that the POS tagger fails to tag 100 percent correctly. Show the input and output. Explain your conjecture as to why the tagger might have been less than perfect with this sentence.
#
#
# 2.	Run a different POS tagger in Python. Process the same two sentences from question 1.
#     - a. Does it produce the same or different output?
#     - b. Explain any differences as best you can.
#
#
# 3.	In a news article from this week’s news, find a random sentence of at least 10 words.
#     - a. Looking at the Penn tag set, manually POS tag the sentence yourself.
#     - b. Now run the same sentences through both taggers that you implemented for questions 1 and 2. Did either of the taggers produce the same results as you had created manually?
#     - c. Explain any differences between the two taggers and your manual tagging as much as you can.
#
#
# %%
# python
from collections import defaultdict

# nltk
import nltk
from nltk.tokenize import ToktokTokenizer

# spacy
import spacy

# %%
nltk.download("averaged_perceptron_tagger")
sp = spacy.load("en_core_web_sm")
# %%
##### Global Variables #####
toktok = ToktokTokenizer()
# %%
sentence1 = """Similarly, when someone is failing, the tendency is to get on a downward spiral that can even become a self-fulfilling prophecy."""
sentence2 = """If you really look closely, most overnight successes took a long time."""

loc = locals()

sentences = {key: value for key, value in loc.items() if key.startswith("sentence")}

# tokenizing the sentences - nltk
tokenized_sentences_nltk = {
    name: toktok.tokenize(sentence) for name, sentence in sentences.items()
}

# examining the length of each - nltk tokenized sentence
for name, ts in tokenized_sentences_nltk.items():
    print(
        f"{name} with nltk has {len(ts)} tokens, including punctuation:\n  {ts} \n\n",
        "-" * 40,
        "\n",
    )
# %%
# POS tagging using nltk
pos_tags_nltk = {
    name: nltk.pos_tag(ts) for name, ts in tokenized_sentences_nltk.items()
}

for name, pos_tags in pos_tags_nltk.items():
    print(
        f"{name} has the following nltk POS tags:\n  {pos_tags} \n\n", "-" * 40, "\n",
    )
# %%
# function to extract spacy objects
def sp_pos(sentence):
    return list(zip([str(i) for i in sentence], [i.tag_ for i in sentence]))


# tokenizing the sentences - spacy
tokenized_sentences_spacy = {name: sp(sentence) for name, sentence in sentences.items()}

# examining the length of each - spacy tokenized sentence
for name, ts in tokenized_sentences_spacy.items():
    print(
        f"{name} with spacy has {len(ts)} tokens, including punctuation:\n  {ts} \n\n",
        "-" * 40,
        "\n",
    )
# %%
# POS tagging using spacy
pos_tags_spacy = {name: sp_pos(ts) for name, ts in tokenized_sentences_spacy.items()}

for name, pos_tags in pos_tags_spacy.items():
    print(
        f"{name} has the following spacy POS tags:\n  {pos_tags} \n\n", "-" * 40, "\n",
    )
# %%
# comparing nltk and spacy POS tagging
# finding the same POS tags
same_pos_tags = {
    k: pos_tags_nltk[k]
    for k in pos_tags_nltk
    if k in pos_tags_spacy and pos_tags_nltk[k] == pos_tags_spacy[k]
}

# finding different POS tags
different_pos_tags = {
    k: pos_tags_nltk[k] for k in set(pos_tags_nltk) - set(same_pos_tags)
}

print(
    f"nltk and spacy POS tagged {len(same_pos_tags)} sentence(s) as the same. {list(different_pos_tags.keys())} were not tagged the same."
)
# %%
# function to get difference of two lists so we can see the
def diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


pos_tags = defaultdict(list)

# combining the POS tagging from nltk and spacy to the same dictionary for comparison
for key, value in pos_tags_nltk.items():
    pos_tags[key].append(value)

for key, value in pos_tags_spacy.items():
    pos_tags[key].append(value)

# comparing the POS tagging differences
pos_tags_diff = {k: diff(pos[0], pos[1]) for k, pos in pos_tags.items()}

for name, diffs in pos_tags_diff.items():
    print(
        f"{name} has the following POS tag differences:\n  {diffs} \n\n",
        "-" * 40,
        "\n",
    )
# %% [markdown]
# It is evident that the POS taggers treat words differently for each of the two sentences.
#
# For the first sentence, nltk treats `self-fulfilling` as a single word `('self-fulfilling', 'JJ')`, whereas spacy tags `self`, `-`, `fulfilling` separately as `('self', 'NN'), ('-', 'HYPH'), ('fulfilling', 'VBG')`.
# This can be traced all the way to the tokenizer for each. nltk had 24 tokens, while spacy had 26.in
#
# For the second sentence, treats `look` and `most` as `('look', 'VB'), ('most', 'JJS')` while spacy treats them as `('look', 'VBP'), ('most', 'RBS')`.
# %% [markdown]
# ## Comparing NLTK vs spacy with latest news snippet
#
# News article: https://www.cnn.com/2021/06/15/weather/arizona-smoke-record-temperatures-wildfire/index.html
# %%
news_sentence_manual = [
    ("Wildfire", "NN"),
    ("smoke", "NN"),
    ("could", "MD"),
    ("keep", "VB"),
    ("Phoenix", "NNP"),
    ("from", "IN"),
    ("reaching", "VBG"),
    ("a", "DT"),
    ("record", "NN"),
    ("high", "JJ"),
    ("temperature", "NN"),
    ("Tuesday", "NNP"),
    (".", "."),
]
news_sentence = """Wildfire smoke could keep Phoenix from reaching a record high temperature Tuesday."""

sentences = {"news_sentence": news_sentence}
# %%
# tokenizing the sentences - nltk
tokenized_sentences_nltk_news = {
    name: toktok.tokenize(sentence) for name, sentence in sentences.items()
}

# examining the length of each - nltk tokenized sentence
for name, ts in tokenized_sentences_nltk_news.items():
    print(
        f"{name} with nltk has {len(ts)} tokens, including punctuation:\n  {ts} \n\n",
        "-" * 40,
        "\n",
    )

# tokenizing the sentences - spacy
tokenized_sentences_spacy_news = {
    name: sp(sentence) for name, sentence in sentences.items()
}

# examining the length of each - spacy tokenized sentence
for name, ts in tokenized_sentences_spacy_news.items():
    print(
        f"{name} with spacy has {len(ts)} tokens, including punctuation:\n  {ts} \n\n",
        "-" * 40,
        "\n",
    )
# %%
# POS tagging using nltk
pos_tags_nltk_news = {
    name: nltk.pos_tag(ts) for name, ts in tokenized_sentences_nltk_news.items()
}

for name, pos_tags in pos_tags_nltk_news.items():
    print(
        f"{name} has the following nltk POS tags:\n  {pos_tags} \n\n", "-" * 40, "\n",
    )

# POS tagging using spacy
pos_tags_spacy_news = {
    name: sp_pos(ts) for name, ts in tokenized_sentences_spacy_news.items()
}

for name, pos_tags in pos_tags_spacy_news.items():
    print(
        f"{name} has the following spacy POS tags:\n  {pos_tags} \n\n", "-" * 40, "\n",
    )
#%%
# comparing nltk and spacy POS tagging
# finding the same POS tags
same_pos_tags_news = {
    k: pos_tags_nltk_news[k]
    for k in pos_tags_nltk_news
    if k in pos_tags_spacy_news and pos_tags_nltk_news[k] == pos_tags_spacy_news[k]
}

# finding different POS tags
different_pos_tags_news = {
    k: pos_tags_nltk_news[k] for k in set(pos_tags_nltk_news) - set(same_pos_tags_news)
}

print(
    f"nltk and spacy POS tagged {len(same_pos_tags_news)} sentence(s) as the same. {list(different_pos_tags_news.keys())} were not tagged the same."
)
# %%
pos_tags_manual = {"news_sentence": news_sentence_manual}

pos_tags_news = defaultdict(list)

# combining the POS tagging from nltk and spacy and manual to the same dictionary for comparison
for key, value in pos_tags_nltk_news.items():
    pos_tags_news[key].append(value)

for key, value in pos_tags_spacy_news.items():
    pos_tags_news[key].append(value)

# comparing the POS tagging differences between nltk and spacy
pos_tags_diff_news = {k: diff(pos[0], pos[1]) for k, pos in pos_tags_news.items()}

for name, diffs in pos_tags_diff_news.items():
    print(
        f"{name} has the following POS tag differences:\n  {diffs} \n\n",
        "-" * 40,
        "\n",
    )
# %% [markdown]
# Both nltk and spacy tokenize the news sentence as the same, but tag `Wildfire` differently as both `('Wildfire', 'NNP')` from nltk and `('Wildfire', 'NN')` from spacy.
# While spacy treats `Wildfire` as a singular noun, nltk tags it as proper noun. In this case, it's just the beginning of the sentence,
# which is why it's capitalized, but it's not a proper noun.
# %%
pos_tags_news_man_nltk = defaultdict(list)
pos_tags_news_man_spcy = defaultdict(list)

# adding in the manual tagging to compare to nltk and spacy
for key, value in pos_tags_manual.items():
    pos_tags_news_man_nltk[key].append(value)  # add manual to compare to nltk
    pos_tags_news_man_spcy[key].append(value)  # add manual to compare to spacy

for key, value in pos_tags_nltk_news.items():
    pos_tags_news_man_nltk[key].append(value)  # add nltk to compare to manual

for key, value in pos_tags_spacy_news.items():
    pos_tags_news_man_spcy[key].append(value)  # add spacy to compare to manual

# comparing the POS tagging differences
pos_tags_diff_news_man_nltk = {
    k: diff(pos[0], pos[1]) for k, pos in pos_tags_news_man_nltk.items()
}
pos_tags_diff_news_man_spcy = {
    k: diff(pos[0], pos[1]) for k, pos in pos_tags_news_man_spcy.items()
}
# %%
# checking the difference between nltk and manual POS tagging
for name, diffs in pos_tags_diff_news_man_nltk.items():
    print(
        f"{name} has the following POS tag differences between manual and nltk:\n  {diffs} \n\n",
        "-" * 40,
        "\n",
    )

# checking the difference between spacy and manual POS tagging
for name, diffs in pos_tags_diff_news_man_spcy.items():
    print(
        f"{name} has the following POS tag differences between manual and spacy:\n  {diffs} \n\n",
        "-" * 40,
        "\n",
    )

# %% [markdown]
# Because I treated `Wildfire` as a singular noun, similar to spacy, spacy had 0 differences in our POS tagging.
# However, because nltk treats `Wildfire` as a proper noun, we had one difference, as noted above, in our POS tagging.

# %%
