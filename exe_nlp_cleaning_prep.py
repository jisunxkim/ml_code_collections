import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

import string
import re

text = """By the late 1970s, the human potential movement had developed into an industry and provided a market for some NLP ideas. At the center of this growth was the Esalen Institute at Big Sur 12345, California. Perls had led numerous Gestalt therapy seminars at Esalen. Satir was an early leader and Bateson was a guest teacher. Bandler and Grinder claimed that in addition to being a therapeutic method, NLP was also a study of communication and began marketing it as a business tool, claiming that, "if any human being can do anything, so can you."[20] After 150 students paid $1,000 each for a ten-day workshop 2.01 in Santa Cruz, California, Bandler and Grinder gave up academic writing and started producing popular books from seminar transcripts, such as Frogs into Princes, which sold more than 270,000 copies. According to court documents relating to an intellectual property dispute between Bandler and Grinder, Bandler made more than $800,000 in 1980 from workshop and book sales 20%."""

def text_lowercase(text):
    return text.lower()

def text_remove_numbers(text):
    """
    return text after replace ['1970', '12345,', '20', '150', '$1,000', 
    '2.01', '270,000', '$800,000', '1980', '20%'] with a ''
    => remove them from the text 
    """
    num_pattern = r"(?:\$?\d+\.?\,?\d*%?)"
    res = re.sub(num_pattern, '', text)
    return res 

def text_find_numbers(text):    
    """
    return: ['1970', '12345,', '20', '150', '$1,000', 
    '2.01', '270,000', '$800,000', '1980', '20%']
    """
    num_pattern = r"(?:\$?\d+\.?\,?\d*%?)"
    res = re.findall(num_pattern, text)
    return res 

def text_remove_punctuation(text):
    """
    remove all punctuations.
    string.punction => '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    str.maketrans(
        x: string to replace| dictionary, 
        y: Optional[string the same length of x to repalce with], 
        z: Optional[a string describing which characters 
            to remove from the original string])
    translation_table = str.maketrans("abc", "123", "xyz")
    "akbxy".translate(translation_table) => '1k2' 
    if want to remove sring only, use '' for x and y 
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def text_remove_whitespace(text):
    """
    remove white space
    "   A  B   ".split() => ['A', 'B']
    """
    return " ".join(text.split())

def text_remove_stopwords(text):
    """
    should download stopwords:
        nltk.download('stopwords')
    should download punkt to use tokenize 
        nltk.download('punkt')
     
    set(nltk.corpus.stopwords.words("english")) => {'me', 'am', 'be', 'below', 'yours', 'between', 'its', 'who', 'if', 'doesn', 'wasn', 'couldn', 'where', 'above', 'your', 'this', 'should', 'only', 'theirs', "hadn't", 'into', 'been', 'whom', 'being', 'through', 'so', 'myself', 'aren', "should've", 'mightn', 'his', 're', 'yourselves', 'does', "shouldn't", 'now', 'hers', 'o', 'we', 'have', 'can', 'a', 'herself', 'having', 'over', 'they', 'has', 'and', 'it', 'all', "it's", 'to', "weren't", 'ours', 'don', 'of', 'haven', 'in', 'same', 'or', 'him', 'that', 'yourself', 'he', 'once', 'isn', "won't", 'did', 'down', 'm', "mustn't", 'hasn', 'hadn', 'what', "mightn't", 'not', 'before', 'd', 'there', 'after', 'but', "shan't", 'the', "wasn't", 'weren', 'out', 'won', 'mustn', 'just', "hasn't", 'while', 'them', 'my', 'had', 'by', "aren't", 'itself', 'against', 'shouldn', 'some', 'then', 'too', 'do', 'shan', 'up', "isn't", 'from', 'until', 's', 'with', 'such', 'during', 'ain', 'further', 'here', 'off', 'few', 'didn', 'on', 'i', "haven't", 'our', 'ma', 'for', "don't", 'how', 'you', "that'll", 'which', 'why', "doesn't", 'these', 'wouldn', 'about', 't', 'because', 'at', "couldn't", "didn't", 'both', "you're", 'most', 'than', 'ourselves', 'himself', 've', 'each', "you'll", 'more', 'will', 'themselves', 'other', 'y', 'those', 'an', 'no', 'when', "you'd", 'as', 'under', 'was', 'again', 'needn', 'her', 'are', 'their', 'were', 'doing', 'nor', "you've", "she's", "needn't", 'll', "wouldn't", 'very', 'own', 'is', 'any', 'she'}
    """
    stop_words = nltk.corpus.stopwords.words("english")
    
    if isinstance(text, str):
        # Tokenize the string into words
        # words = nltk.word_tokenize(text)
        words = text.split()
        # Remove stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words] 
        # Join the filtered words back into a string
        text = ' '.join(filtered_words)
    elif isinstance(text, list):
        # Remove stopwords from the list of words
        text = [word for word in text if word.lower() not in stop_words]
    
    return text
    
def text_stem_words(text):
    """
    Stemming is the process of getting the root form of a word. Stem or root is the part to which inflectional affixes (-ed, -ize, -de, -s, etc.) are added. The stem of a word is created by removing the prefix or suffix of a word. So, stemming a word may not result in actual words.
    books      --->    book
    looked     --->    look
    denied     --->    deni
    flies      --->    fli

    """
    stemmer = nltk.stem.PorterStemmer()
    
    if isinstance(text, str):
        words = text.split()
        stem_words = [stemmer.stem(word) for word in words]
        return ' '.join(stem_words)
    elif isinstance(text, list):
        return  [stemmer.stem(word) for word in words]

def text_lemmatization(text):
    """
    Should download 'wordnet'
        nltk.download('wordnet')
    Like stemming, lemmatization also converts a word to its root form. The only difference is that lemmatization ensures that the root word belongs to the language. We will get valid words if we use lemmatization.
    example: 
        Input: data science uses scientific methods algorithms and many types of processes 
        Output: [data, science, use, scientific, methods, algorithms, and, many, type, of, process] 
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    if isinstance(text, str):
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    elif isinstance(text, list): 
        return [lemmatizer.lemmatize(word) for word in text]
    

print("before NL data cleaning...:")
print(text)
text = text_lowercase(text)
text = text_remove_numbers(text)
text = text_remove_punctuation(text)
text = text_remove_whitespace(text)
text = text_remove_stopwords(text)
text = text_lemmatization(text)
print("="*20)
print("after NL data cleaning...:")
print(text)            


