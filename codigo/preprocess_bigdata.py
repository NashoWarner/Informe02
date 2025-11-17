from datasets import load_dataset
from transformers import AutoTokenizer

class Preprocess:
    def __init__(self):
        '''
        Initialize the dataset and tokenizer for preprocessing Wikipedia data.
        
        For the dataset:
        - Load the 'wikipedia' dataset with '20220301.en' configuration
        - Enable streaming=True to handle the large 20GB dataset
        
        For the tokenizer:
        - Use 'distilbert-base-uncased' pretrained tokenizer
        '''
        # Load Wikipedia dataset with streaming enabled
        self.dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True, trust_remote_code=True)
        
        # Initialize DistilBert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize(self, examples, max_length=100):
        '''
        Tokenize the text from examples.
        
        Args:
            examples: Dictionary containing 'text' key with article texts
            max_length: Maximum number of tokens to keep (default: 100)
        
        Return:
            Dictionary with tokenized data containing only the first max_length tokens
        '''
        # Tokenize the text with truncation
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
    
        # Return only input_ids as 'tokens'
        return {'tokens': tokenized['input_ids']}
    
    def preprocess_text(self, batch_size=1000):
        '''
        Preprocess the dataset by tokenizing articles in batches.
        
        Args:
            batch_size: Number of examples to process at once (default: 1000)
        
        Return:
            Preprocessed dataset with id, title, and first 100 tokens
        '''
        # Apply tokenization in batches
        tokenized_dataset = self.dataset.map(
            lambda examples: self.tokenize(examples, max_length=100),
            batched=True,
            batch_size=batch_size,
            remove_columns=['text', 'url']  # Remove original text and url to save memory
        )
        
        return tokenized_dataset