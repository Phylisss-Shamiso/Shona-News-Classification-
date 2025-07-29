from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class EmbeddingGenerator(BaseEstimator, TransformerMixin):
    """Generates embeddings using your custom model and tokenizer"""

    def __init__(self, model_path, tokenizer_path, max_length=512):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def _load_components(self):
        """Lazy loading of tokenizer and model"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.eval()

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        self._load_components()
        if isinstance(X, pd.DataFrame):
            # Handle DataFrame input
            texts = X.iloc[:, 0].tolist()
        else:
            # Handle list/array input
            texts = X

        tokenized_inputs = [
            self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            for text in texts
        ]

        all_embeddings = []
        for tokens in tokenized_inputs:
            with torch.no_grad():
                inputs = {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask']
                }
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state

            # Process embeddings exactly as in your original code
            embeddings = last_hidden_state.squeeze(0)  # Remove batch dimension
            sequence_embedding = embeddings.mean(dim=0).detach().numpy()
            all_embeddings.append(sequence_embedding)

        return np.array(all_embeddings)
