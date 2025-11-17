import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

class Preprocess(object):
	def __init__(self):
		pass

	def clean_text(self, text):
		#1 limpia html
		cleaned_text = re.sub(r'<[^>]+>', '', text)
		cleaned_text = re.sub(r'&[a-z]+;', '', text)
		# Step 2 is implemented for you
		cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+$',' ',cleaned_text).strip()
		#3 nada
		#4 minuscula
		cleaned_text = cleaned_text.lower()
		#5 tokeinzador
		stop_works = set(stopwords.words("english"))
		tokens = word_tokenize(cleaned_text)
		tokens = [word for word in tokens if word not in stop_works]
		#6 lo junta
		cleaned_text = " ".join(tokens)

		return cleaned_text

	def clean_dataset(self, data):
		# Limpia las cadenas
		return [self.clean_text(text) for text in data]
		
		


def clean_wos(x_train, x_test):
	preprocessor = Preprocess()
	clean_text_wos = preprocessor.clean_dataset(x_train)
	clean_text_wos_test = preprocessor.clean_dataset(x_test)
	return clean_text_wos, clean_text_wos_test
