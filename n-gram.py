import re
from collections import defaultdict
import heapq

class NGramAutocomplete:
    def __init__(self, n=3):
        """
        Initialize the autocomplete system
        :param n: N-gram size (default is trigram)
        """
        self.n = n
        # Store n-gram frequencies
        self.ngram_freq = defaultdict(lambda: defaultdict(int))
        # Store word frequencies for unigram suggestions
        self.word_freq = defaultdict(int)
        
    def preprocess_text(self, text):
        """
        Clean and tokenize input text
        :param text: Input text string
        :return: List of cleaned tokens
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text.split()
    
    def train(self, corpus):
        """
        Train the autocomplete model on a given corpus
        :param corpus: List of text documents or single large text
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        
        for document in corpus:
            tokens = self.preprocess_text(document)
            
            # Update unigram frequencies
            for word in tokens:
                self.word_freq[word] += 1
            
            # Generate n-grams
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                next_word = tokens[i+self.n-1]
                self.ngram_freq[context][next_word] += 1
    
    def get_suggestions(self, partial_text, top_k=5):
        """
        Generate autocomplete suggestions
        :param partial_text: Partial text to complete
        :param top_k: Number of top suggestions to return
        :return: List of suggested completions
        """
        tokens = self.preprocess_text(partial_text)
        
        # If not enough context, use unigram frequencies
        if len(tokens) < self.n - 1:
            candidates = sorted(
                self.word_freq.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return [word for word, _ in candidates[:top_k]]
        
        # Use last n-1 tokens as context
        context = tuple(tokens[-(self.n-1):])
        
        # Collect suggestions based on n-gram frequencies
        candidates = self.ngram_freq[context]
        
        # Use heap to get top-k suggestions
        top_suggestions = heapq.nlargest(
            top_k, 
            candidates.items(), 
            key=lambda x: x[1]
        )
        
        return [word for word, _ in top_suggestions]

# Example usage
def main():
    # Sample training corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "python is a great programming language for machine learning",
        "natural language processing involves understanding human language",
        "machine learning algorithms can predict text completions"
    ]

    # Create and train the autocomplete system
    autocomplete = NGramAutocomplete(n=3)
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

if __name__ == "__main__":
    main()