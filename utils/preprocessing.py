import re
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Kamus normalisasi
kamus_df = pd.read_csv(
    "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv",
    header=None, names=["tidak_baku", "baku"], encoding="ISO-8859-1"
)
normalization_dict = dict(zip(kamus_df["tidak_baku"], kamus_df["baku"]))

# Stopwords bawaan + tambahan
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stopwords.update({
    "ovo", "nya", "mau", "sangat", "banget", "terus", "kalau", "malah", "padahal", "bagaimana", "sama",
    "apa", "gue", "kok", "sih", "jadi", "kali", "nih", "kayak", "lah", "cuma", "mana", "dong", "pas",
    "kena", "aku", "masa", "x", "paling", "kan", "memang", "kamu", "enggak", "kalian"
})

stemmer = StemmerFactory().create_stemmer()

# Fungsi Preprocessing
def full_preprocessing(text):
    # Text Cleaning
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Case folding
    text = text.lower()

    # Tokenizing
    tokens = text.split()  # Bisa gunakan word_tokenize jika ingin

    # Normalization
    tokens = [normalization_dict.get(word, word) for word in tokens]

    # Gabungkan lagi untuk phrasing
    text = ' '.join(tokens)

    # Pre-processing frasa
    text = re.sub(r'\btop up\b', 'topup', text, flags=re.IGNORECASE)
    text = re.sub(r'\be wallet\b', 'ewallet', text, flags=re.IGNORECASE)

    # Stopword removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    cleaned = ' '.join(tokens)

    # Stemming
    return stemmer.stem(cleaned)