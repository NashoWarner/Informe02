from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict


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
        self.dataset = load_dataset(
            'wikipedia', 
            "20220301.en", 
            split='train', 
            streaming=True,
            trust_remote_code=True  # FIX para el ValueError
        )

        # Initialize DistilBert tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    def tokenize(self, batch, max_length=100) -> Dict[str, List[List[str]]]:
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
            batch['text'],
            max_length=max_length,
            padding='max_length',  
            truncation=True       
        )
        tokens_list = [self.tokenizer.convert_ids_to_tokens(ids)for ids in tokenized['input_ids']]

        return {"tokens": tokens_list}

    def preprocess_text(self):
        '''
        Preprocess the dataset by tokenizing articles in batches.
        
        Args:
            batch_size: Number of examples to process at once (default: 1000)
        
        Return:
            Preprocessed dataset with id, title, and first 100 tokens
        '''
        dataset_with_tokens = self.dataset.map(
            self.tokenize,
            batched=True,
            batch_size=1000
        )
        dataset_cleaned = dataset_with_tokens.select_columns(['id', 'title', 'tokens'])

        return dataset_cleaned