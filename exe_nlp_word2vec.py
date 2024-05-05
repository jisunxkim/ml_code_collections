from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Sample sentences
sentences = [
    'I enjoy listening to music',
    'Music is my passion',
    'I like playing the guitar',
    'I am learning to play piano',
    'I love classical music'
]

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenize sentences and lemmatize
tokenized_sentences = []
for sentence in sentences:
    words = word_tokenize(sentence.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    tokenized_sentences.append(lemmatized_words)

# Generate n-grams
ngram_sentences = []
for sentence in tokenized_sentences:
    unigrams = list(sentence)
    bigrams = list(ngrams(sentence, 2))
    trigrams = list(ngrams(sentence, 3))
    ngram_sentences.append(unigrams + bigrams + trigrams)

# Train the Word2Vec model
model = Word2Vec(ngram_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the word vector for a particular word
word_vector = model.wv['music']
print("Vector representation of 'music':", word_vector)

# Find similar words
similar_words = model.wv.most_similar('music')
print("Words similar to 'music':", similar_words)