import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
import string
def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)

    preprocessed_text = text.lower()   # 將輸入都轉成小寫
    preprocessed_text = "".join([char for char in preprocessed_text if (char not in string.punctuation)]) # 把標點符號刪掉
    stopword_list = stopwords.words('english') # 把所有stopwords叫出來
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(preprocessed_text) # 把text變成 token
    tokens = [word for word in tokens if word not in stopword_list]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]  # 把每個token都做stemming
    preprocessed_text = ' '.join(stemmed)  # join回preprocessed_text
    preprocessed_text = preprocessed_text.replace("<br />", " ")   # 刪掉換行符號


    return preprocessed_text



