from datasets import load_dataset 
from transformers import AutoTokenizer

### SUBMIT THIS FILE TO THE BONUS EXTRA CREDIT GRADESCOPE SECTION ###

class Preprocess:
    def __init__(self):
        '''
        Inicialice el conjunto de datos y el tokenizador.

        Para self.dataset, se proporciona la mayor parte del código. Estamos importando el conjunto de datos de Wikipedia desde Hugging Face.
        Sin embargo, debido al contexto de big data (¡el tamaño del conjunto de datos es de 20 GB!), necesitaremos modificarlo para permitir la transmisión (https://huggingface.co/docs/datasets/stream) de modo que no carguemos todo en la
        memoria. Tenga en cuenta que después de permitir la transmisión, self.dataset será del tipo IterableDataset, cuya documentación se puede encontrar aquí:
        https://huggingface.co/docs/datasets/v2.14.4/en/package_reference/main_classes#datasets.IterableDataset.
        También puede intentar ejecutar el código una vez tal como está para comparar la transmisión con la no transmisión y ver por qué es tan importante.

        Para self.tokenizer, usaremos el AutoTokenizer que importamos de la biblioteca de transformadores Hugging Face.
        Usaremos el tokenizador 'distilbert-base-cased', que es una versión simplificada del LLM, BERT. Aprenderemos más sobre el funcionamiento interno de BERT más adelante en el curso.


        Initialize the dataset and the tokenizer. 

        For self.dataset, most of the code is given. We are importing the Wikipedia dataset from Hugging Face.
        However, because of the big data context (size of the dataset is 20GB!), we will need to alter it to 
        allow for streaming (https://huggingface.co/docs/datasets/stream) so that we don't load all of it into
        memory. Note that after allowing for streaming, self.dataset will be of the type IterableDataset, 
        documentation of which can be found here: 
        https://huggingface.co/docs/datasets/v2.14.4/en/package_reference/main_classes#datasets.IterableDataset.
        You could also try running the code once as is to compare streaming vs non-streaming, and see why it is 
        so important!

        For self.tokenizer, let's use the AutoTokenizer we imported from the Hugging Face transformers library. 
        We will use the 'distilbert-base-cased' tokenizer, which is a distilled version of the LLM, BERT. We will
        be learning more about the inner-workings of BERT later on in the course.

        Autotokenizer: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
        Distilbert: https://huggingface.co/distilbert-base-cased
        Tokenizer: https://huggingface.co/docs/transformers/main_classes/tokenizer
        '''
        #5.1.1
        self.dataset = load_dataset('wikipedia', "20220301.en", split='train')

        #5.1.2
        self.tokenizer = None

    def tokenize(self, batch, max_length=100):
        '''
        Este es un método auxiliar que será llamado por preprocess_text() para convertir en tokens la columna 'text' en lotes.
        Tenga en cuenta que en este método no solo estamos convirtiendo en tokens el texto usando Distilbert, sino que también estamos convirtiendo
        esos identificadores de token numéricos nuevamente en tokens en inglés. La documentación para los tokenizadores se puede encontrar aquí:
        https://huggingface.co/docs/transformers/main_classes/tokenizer

        Para el tokenizador, siempre queremos que las matrices de tokens tengan un tamaño max_length. Asegúrese de configurar las estrategias de relleno y truncamiento para permitir esto.

        This is a helper method that will be called by preprocess_text() to tokenize the 'text' column in batch. 
        Keep in mind that in this method we are not just tokenizing the text using Distilbert, we are then converting 
        those numerical token ids back to English tokens. Documentation for tokenizers can be found here: 
        https://huggingface.co/docs/transformers/main_classes/tokenizer

        For the tokenizer, we always want the arrays of tokens to be size max_length. Make sure to set the padding and
        truncation strategies to allow for this.

        Args:
            batch: group of samples streamed from the dataset
            max_length: maximum length of the tokenized text to pad/truncate to
        Return:
            tokens_dict: dictionary mapping the word 'tokens' to a list of English tokens for each sample in the batch. 
                         This output will then be used in the preprocess_text() method to add a new 'tokens' column to 
                         self.dataset.
        '''
        # 5.1.2
        tokens_dict = None
        return tokens_dict
    
    def preprocess_text(self):
        '''
        Aquí, aplicaremos el método self.tokenize en self.dataset. Recuerde conservar solo las columnas 'id', 'title' y
        'tokens'. Asegúrese de utilizar el procesamiento por lotes (use un lote de 1000) para permitir una velocidad de procesamiento adecuada.

        La función map puede resultar útil.

        Here, we will apply the self.tokenize method on self.dataset. Remember to only keep the columns 'id', 'title', and 
        'tokens'. Make sure to use batching (use batch of 1000) to allow for proper processing speed. 

        The map function may be useful.

        Return:
            dataset_cleaned: Iterable Dataset with the 'id', 'title', and new 'tokens' column added
        '''
        #5.1.3
        dataset_cleaned = None
        return dataset_cleaned
    

    


