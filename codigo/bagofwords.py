import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Función global para ProcessPoolExecutor (fuera de la clase)
def _process_single_document(words_and_encoder):
    """
    Función auxiliar global para procesar un documento en paralelo
    """
    words, encoder_categories, vocab_size = words_and_encoder
    
    try:
        if len(words) == 0:
            return np.zeros((1, vocab_size))
        else:
            # Recrear OneHotEncoder temporal con las categorías
            temp_oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            temp_oh.fit(encoder_categories.reshape(-1, 1))
            
            # Transformar palabras
            words_array = np.array(words).reshape(-1, 1)
            onehot_encoded = temp_oh.transform(words_array)
            bow_vector = np.sum(onehot_encoded, axis=0).reshape(1, -1)
            return bow_vector
    except:
        return np.zeros((1, vocab_size))


class OHE_BOW(object): 
    def __init__(self):
        '''
        Initialize instance of OneHotEncoder in self.oh for use in fit and transform
        '''
        self.vocab_size = None	#keep
        self.oh = OneHotEncoder(handle_unknown='ignore')  # Ignorar palabras desconocidas

    def split_text(self, data):
        '''
        Helper function to separate each string into a list of individual words
        Args:
            data: list of N strings
        
        Return:
            data_split: list of N lists of individual words from each string
        '''
        data_split = [text.split() for text in data]
        return data_split

    def flatten_list(self, data):
        '''
        Helper function to flatten a list of list of words into a single list
        Args:
            data: list of N lists of W_i words 
        
        Return:
            data_split: (W,) numpy array of words, 
                where W is the sum of the number of W_i words in each of the list of words	
        '''
        flattened = []
        for word_list in data:
            flattened.extend(word_list)
        return np.array(flattened)

    def fit(self, data):
        '''
        Fit the initialized instance of OneHotEncoder to the given data
        Use split_text to separate the given strings into a list of words and 
        flatten_list to flatten the list of words in a sentence into a single list of words
        
        Set self.vocab_size to the number of unique words in the given data corpus
        Args:
            data: list of N strings 
        
        Return:
            None
        Hint: You may find numpy's reshape function helpful when fitting the encoder
        '''
        # Separar el texto en palabras
        data_split = self.split_text(data)
        
        # Aplanar la lista de palabras
        flattened_words = self.flatten_list(data_split)
        
        # Obtener el tamaño del vocabulario (palabras únicas)
        self.vocab_size = len(np.unique(flattened_words))
        
        # Reshape para fit del OneHotEncoder (necesita formato columna)
        flattened_reshaped = flattened_words.reshape(-1, 1)
        
        # Ajustar el OneHotEncoder
        self.oh.fit(flattened_reshaped)

    def onehot(self, words):
        '''
        Helper function to encode a list of words into one hot encoding format
        Args:
            words: list of W_i words from a string
        
        Return:
            onehotencoded: (W_i, D) numpy array where:
                W_i is the number of words in the current input list i
                D is the vocab size
        Hint: 	.toarray() may be helpful in converting a sparse matrix into a numpy array
                You can use sklearn's built-in OneHotEncoder transform function
        '''
        # Convertir lista de palabras a numpy array y reshape
        words_array = np.array(words).reshape(-1, 1)
        
        # Transformar usando OneHotEncoder
        onehotencoded = self.oh.transform(words_array).toarray()
        
        return onehotencoded
    
    def transform(self, data):
        '''
        Use the already fitted instance of OneHotEncoder to help you transform the given 
        data into a bag of words representation. You will need to separate each string 
        into a list of words and iterate through each list to transform into a one hot 
        encoding format.
        Use your one hot encoding of each word in a sentence to get the bag of words count
        representation. You may want to look 
        For any empty strings append a (1, D) array of zeros instead.
            
        Args:
            data: list of N strings
        
        Return:
            bow: (N, D) numpy array
        Hint: Using a try and except block during one hot encoding transform may be helpful
        '''
        # Separar el texto en palabras
        data_split = self.split_text(data)
        
        # Preparar datos para paralelización
        encoder_categories = self.oh.categories_[0]
        tasks = [(words, encoder_categories, self.vocab_size) for words in data_split]
        
        # Usar ProcessPoolExecutor para paralelizar
        with ProcessPoolExecutor() as executor:
            bow_list = list(executor.map(_process_single_document, tasks))
        
        # Concatenar todos los vectores
        bow = np.vstack(bow_list)
        
        return bow