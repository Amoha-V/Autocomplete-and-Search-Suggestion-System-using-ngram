import re
import pickle
import os
from collections import defaultdict
import heapq
import math
import random
import numpy as np

class NGramAutocomplete:
    def __init__(self, n=3, smoothing=None, discount=0.75):
        """
        Initialize the N-gram autocomplete system with advanced features
        
        Parameters:
        -----------
        n : int
            N-gram size (default is trigram)
        smoothing : str or None
            Smoothing method: 'kneser_ney', 'laplace', or None
        discount : float
            Discount parameter for Kneser-Ney smoothing (0.1-0.9)
        """
        self.n = n
        self.smoothing = smoothing
        self.discount = discount
        
        # N-gram frequency counters
        self.ngram_freq = defaultdict(lambda: defaultdict(int))  # Context -> next word -> count
        self.ngram_minus1_freq = defaultdict(int)  # Count of contexts
        
        # Word frequency stats
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.vocab_size = 0
        
        # For Kneser-Ney continuation count
        self.continuation_count = defaultdict(int)  # How many contexts a word follows
        self.total_continuations = 0  # Total continuation count
        
        # Training state
        self.is_trained = False
        
    def preprocess_text(self, text):
        """
        Clean and tokenize input text
        
        Parameters:
        -----------
        text : str
            Input text string
            
        Returns:
        --------
        list
            List of cleaned tokens
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text.split()
    
    def train(self, corpus):
        """
        Train the autocomplete model on a given corpus
        
        Parameters:
        -----------
        corpus : list or str
            List of text documents or single large text
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        
        # Reset counters
        self.ngram_freq = defaultdict(lambda: defaultdict(int))
        self.ngram_minus1_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.continuation_count = defaultdict(int)
        self.total_continuations = 0
        
        # Process each document in the corpus
        for document in corpus:
            tokens = self.preprocess_text(document)
            
            # Add sentence boundary markers for better prediction at start of sentences
            tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            
            # Update unigram frequencies
            for word in tokens:
                self.word_freq[word] += 1
                self.total_words += 1
            
            # Generate n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                
                # Update n-gram frequency counts
                self.ngram_freq[context][next_word] += 1
                self.ngram_minus1_freq[context] += 1
                
                # For Kneser-Ney: track continuation patterns
                # A unique context-word pair counts as 1 continuation
                if self.ngram_freq[context][next_word] == 1:
                    self.continuation_count[next_word] += 1
                    self.total_continuations += 1
        
        # Update vocabulary size
        self.vocab_size = len(self.word_freq)
        self.is_trained = True
        
    def get_suggestions(self, text, top_k=5):
        """
        Generate autocomplete suggestions with probabilities
        
        Parameters:
        -----------
        text : str
            Partial text to complete
        top_k : int
            Number of top suggestions to return
            
        Returns:
        --------
        list of tuples
            List of (word, probability) pairs
        """
        if not self.is_trained:
            return []
            
        tokens = self.preprocess_text(text)
        
        # Get the appropriate context
        if len(tokens) >= self.n - 1:
            context = tuple(tokens[-(self.n-1):])
        else:
            # Padding for short inputs
            padding = ["<s>"] * (self.n - 1 - len(tokens))
            context = tuple(padding + tokens)
        
        # Get probability distribution for next words
        next_word_probs = {}
        
        # Apply appropriate smoothing method
        if self.smoothing == "kneser_ney":
            next_word_probs = self._kneser_ney_prediction(context)
        elif self.smoothing == "laplace":
            next_word_probs = self._laplace_prediction(context)
        else:
            next_word_probs = self._basic_prediction(context)
        
        # Return top k suggestions
        return heapq.nlargest(top_k, next_word_probs.items(), key=lambda x: x[1])
    
    def _basic_prediction(self, context):
        """
        Basic MLE prediction with backoff to shorter contexts
        
        Parameters:
        -----------
        context : tuple
            The context (n-1 previous words)
            
        Returns:
        --------
        dict
            Dictionary of {word: probability}
        """
        result = {}
        
        # If the context exists in our model
        if self.ngram_minus1_freq[context] > 0:
            total_count = self.ngram_minus1_freq[context]
            
            # Calculate probability for each possible next word
            for word, count in self.ngram_freq[context].items():
                result[word] = count / total_count
                
            return result
        
        # Backoff to shorter context if this context isn't found
        if len(context) > 1:
            return self._basic_prediction(context[1:])
        
        # Fallback to unigram (word frequency) model
        for word, count in self.word_freq.items():
            result[word] = count / self.total_words
            
        return result
    
    def _laplace_prediction(self, context):
        """
        Laplace (add-1) smoothed prediction
        
        Parameters:
        -----------
        context : tuple
            The context (n-1 previous words)
            
        Returns:
        --------
        dict
            Dictionary of {word: probability}
        """
        result = {}
        
        # If the context exists in our model
        if self.ngram_minus1_freq[context] > 0:
            total_count = self.ngram_minus1_freq[context]
            
            # Calculate smoothed probability for each word in vocabulary
            for word in self.word_freq.keys():
                count = self.ngram_freq[context][word]
                # Laplace smoothing adds 1 to count and vocab_size to denominator
                result[word] = (count + 1) / (total_count + self.vocab_size)
                
            return result
        
        # Backoff to shorter context if this context isn't found
        if len(context) > 1:
            return self._laplace_prediction(context[1:])
        
        # Fallback to smoothed unigram model
        for word, count in self.word_freq.items():
            result[word] = (count + 1) / (self.total_words + self.vocab_size)
            
        return result
    
    def _kneser_ney_prediction(self, context):
        """
        Kneser-Ney smoothed prediction
        
        Parameters:
        -----------
        context : tuple
            The context (n-1 previous words)
            
        Returns:
        --------
        dict
            Dictionary of {word: probability}
        """
        result = {}
        
        # If the context exists in our model
        if self.ngram_minus1_freq[context] > 0:
            total_count = self.ngram_minus1_freq[context]
            
            # Number of unique words following this context (used for lambda calculation)
            unique_next_words = len(self.ngram_freq[context])
            
            # Calculate lambda (weight for lower-order model)
            lambda_factor = (self.discount * unique_next_words) / total_count
            
            # For each word in our vocabulary
            for word in self.word_freq.keys():
                # Get higher-order probability (with discounting)
                count = self.ngram_freq[context][word]
                higher_order = max(count - self.discount, 0) / total_count
                
                # Get continuation probability (for lower-order model)
                if self.total_continuations > 0:
                    # How many different contexts this word follows
                    continuation_prob = self.continuation_count[word] / self.total_continuations
                else:
                    continuation_prob = 1 / self.vocab_size
                
                # Interpolate higher and lower order models
                result[word] = higher_order + lambda_factor * continuation_prob
                
            return result
        
        # Backoff to lower-order model
        # For simplicity, we'll use continuation counts directly if context not found
        if self.total_continuations > 0:
            for word in self.word_freq.keys():
                result[word] = self.continuation_count[word] / self.total_continuations
        else:
            # Uniform distribution as fallback
            for word in self.word_freq.keys():
                result[word] = 1 / self.vocab_size
                
        return result
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Convert defaultdict to dict for better serialization
            model_data = {
                'n': self.n,
                'smoothing': self.smoothing,
                'discount': self.discount,
                'ngram_freq': dict(self.ngram_freq),
                'ngram_minus1_freq': dict(self.ngram_minus1_freq),
                'word_freq': dict(self.word_freq),
                'total_words': self.total_words,
                'vocab_size': self.vocab_size,
                'continuation_count': dict(self.continuation_count),
                'total_continuations': self.total_continuations,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model attributes
            self.n = model_data['n']
            self.smoothing = model_data['smoothing']
            self.discount = model_data['discount']
            
            # Convert dict back to defaultdict
            self.ngram_freq = defaultdict(lambda: defaultdict(int))
            for context, word_counts in model_data['ngram_freq'].items():
                for word, count in word_counts.items():
                    self.ngram_freq[context][word] = count
            
            self.ngram_minus1_freq = defaultdict(int)
            for context, count in model_data['ngram_minus1_freq'].items():
                self.ngram_minus1_freq[context] = count
            
            self.word_freq = defaultdict(int)
            for word, count in model_data['word_freq'].items():
                self.word_freq[word] = count
            
            self.continuation_count = defaultdict(int)
            for word, count in model_data['continuation_count'].items():
                self.continuation_count[word] = count
            
            self.total_words = model_data['total_words']
            self.vocab_size = model_data['vocab_size']
            self.total_continuations = model_data['total_continuations']
            self.is_trained = model_data['is_trained']
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate_perplexity(self, test_corpus):
        """
        Evaluate model using perplexity (lower is better)
        
        Parameters:
        -----------
        test_corpus : list
            List of sentences to evaluate on
            
        Returns:
        --------
        float
            Perplexity score
        """
        if not self.is_trained:
            return float('inf')
        
        log_probability = 0.0
        total_tokens = 0
        
        for sentence in test_corpus:
            tokens = self.preprocess_text(sentence)
            tokens = ["<s>"] * (self.n - 1) + tokens
            
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                
                # Get probability of this word given context
                if self.smoothing == "kneser_ney":
                    probs = self._kneser_ney_prediction(context)
                elif self.smoothing == "laplace":
                    probs = self._laplace_prediction(context)
                else:
                    probs = self._basic_prediction(context)
                
                prob = probs.get(word, 1e-10)  # Avoid log(0)
                log_probability += math.log2(prob)
                total_tokens += 1
        
        # Calculate perplexity
        if total_tokens == 0:
            return float('inf')
        
        return 2 ** (-log_probability / total_tokens)
    
    def generate_text(self, seed_text="", max_length=20):
        """
        Generate text using the trained model
        
        Parameters:
        -----------
        seed_text : str
            Starting text for generation
        max_length : int
            Maximum number of words to generate
            
        Returns:
        --------
        str
            Generated text
        """
        if not self.is_trained:
            return ""
        
        # Initialize with seed text or start symbol
        if seed_text:
            tokens = self.preprocess_text(seed_text)
        else:
            tokens = ["<s>"] * (self.n - 1)
        
        # Generate words
        for _ in range(max_length):
            # Get context from last n-1 tokens
            context = tuple(tokens[-(self.n-1):])
            
            # Get probability distribution for next words
            if self.smoothing == "kneser_ney":
                next_word_probs = self._kneser_ney_prediction(context)
            elif self.smoothing == "laplace":
                next_word_probs = self._laplace_prediction(context)
            else:
                next_word_probs = self._basic_prediction(context)
            
            # Convert to list for random sampling
            words, probs = zip(*next_word_probs.items()) if next_word_probs else ([""], [1.0])
            
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
            else:
                probs = [1.0/len(probs)] * len(probs)
            
            # Sample next word based on probabilities
            next_word = np.random.choice(words, p=probs)
            
            # Stop if end of sentence
            if next_word == "</s>":
                break
                
            tokens.append(next_word)
        
        # Convert tokens back to text
        generated_text = " ".join([t for t in tokens if t not in ["<s>", "</s>"]])
        return generated_text

# Example usage
if __name__ == "__main__":
    # Simple test
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "python is a great programming language for machine learning",
        "natural language processing involves understanding human language",
        "machine learning algorithms can predict text completions"
    ]
    
    # Create and train the autocomplete system
    autocomplete = NGramAutocomplete(n=3, smoothing="kneser_ney")
    autocomplete.train(corpus)
    
    # Test autocomplete
    test_inputs = [
        "the quick",
        "machine learning",
        "python is"
    ]
    
    print("Autocomplete Suggestions:")
    for input_text in test_inputs:
        suggestions = autocomplete.get_suggestions(input_text)
        print(f"Input: '{input_text}' -> Suggestions: {suggestions}")
    
    # Generate some text
    print("\nGenerated text:")
    print(autocomplete.generate_text("the quick", max_length=10))