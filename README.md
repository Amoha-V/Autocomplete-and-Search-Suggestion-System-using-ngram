
 # Autocomplete and Search Suggestion System using N-grams

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autocomplete-and-search-suggestion-system-using-ngram-6y5pemgk.streamlit.app/)

This project demonstrates the implementation of an **Autocomplete System** based on an n-gram model. It predicts word completions and suggestions using trained n-grams from a given text corpus. This project was inspired by the [Google Books Ngram Viewer](https://books.google.com/ngrams/info).


## Live Demo

Try the live application: [Autocomplete and Search Suggestion System](https://autocomplete-and-search-suggestion-system-using-ngram-6y5pemgk.streamlit.app/)

## Features

- **Advanced N-gram Models**: Configurable N-gram size (2-gram, 3-gram, etc.)
- **Multiple Smoothing Techniques**:
  - Kneser-Ney smoothing
  - Laplace (add-one) smoothing
  - Basic MLE with backoff
- **Interactive Prediction**: Real-time suggestions as you type
- **Text Generation**: Create coherent text using the trained model
- **Model Evaluation**: Calculate perplexity to measure model quality
- **Persistence**: Save and load trained models
- **User-friendly Interface**: Built with Streamlit for easy interaction
### Key Components:

#### 1. Reading and Preprocessing Data:
- The code processes the text input by:
   - Converting the text to **lowercase**.
   - Removing **special characters** using regular expressions.
   - Tokenizing the text into words for further analysis.

#### 2. Creating the N-Gram Model:
- The system uses **n-grams** (default is **trigrams**) to analyze word sequences.
- It performs the following tasks:
   - Builds n-grams using a sliding window approach over the input text.
   - Stores the frequency of each **context-to-next-word** relationship.
   - Tracks **unigram word frequencies** for cases with insufficient context.

#### 3. Generating Autocomplete Suggestions:
- The `get_suggestions` function provides **autocomplete predictions**:
   - For input with **sufficient context**, suggestions are based on the most frequent n-grams.
   - If context is limited (e.g., only one word), it falls back to unigram word frequencies.
- Results are ranked and returned based on their frequency in the training corpus.

### Example Workflow:

1. **Training the Model**:
   - Input text corpus:
     ```python
     corpus = [
         "the quick brown fox jumps over the lazy dog",
         "python is a great programming language",
         "machine learning algorithms can predict text completions"
     ]
     ```
   - The model trains on the corpus to build its n-gram frequency dictionaries.

2. **Getting Suggestions**:
   - Input partial text for autocomplete:
     ```python
     suggestions = autocomplete.get_suggestions("the quick")
     print(suggestions)
     ```

3. **Example Output**:
   ```plaintext
   Input: 'the quick'
   Suggestions: ['brown', 'red', 'fox']

   Input: 'machine learning'
   Suggestions: ['algorithms', 'models', 'techniques']

   Input: 'python is'
   Suggestions: ['a', 'the', 'an']
   ```

---

### Code Execution Steps:
1. **Train the Model**:
   - Train the model using a text corpus by calling:
     ```python
     autocomplete.train(corpus)
     ```

2. **Get Predictions**:
   - Provide input text to fetch autocomplete suggestions:
     ```python
     suggestions = autocomplete.get_suggestions("partial text")
     ```

---

### Customizations:
- **N-Gram Size**:
   - Change the n-gram size by setting the `n` parameter during initialization:
     ```python
     autocomplete = NGramAutocomplete(n=4)  # For 4-grams
     ```
- **Top-K Suggestions**:
   - Control the number of suggestions returned:
     ```python
     autocomplete.get_suggestions("partial text", top_k=10)
     ```

---
## How It Works

The system works by analyzing patterns in text data:

1. **Training**: The model processes text and builds frequency dictionaries for word sequences (N-grams)
2. **Context Analysis**: When given partial input, it identifies the most likely words to follow
3. **Intelligent Prediction**: Uses statistical methods to rank potential completions
4. **Smoothing**: Handles unseen word combinations through advanced smoothing techniques

## Project Structure

```
├── ngram_autocomplete.py     # Core implementation of the N-gram model
├── app.py                    # Streamlit web application
├── requirements.txt          # Dependencies
├── models/                   # Saved model files
├── data/                     # Training corpus
└── README.md                 # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Amoha-V/Autocomplete-and-Search-Suggestion-System-using-ngram.git
cd Autocomplete-and-Search-Suggestion-System-using-ngram

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Usage

### Training a Model

```python
from ngram_autocomplete import NGramAutocomplete

# Create an instance with desired parameters
autocomplete = NGramAutocomplete(n=3, smoothing="kneser_ney")

# Train on your text corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "python is a great programming language for machine learning",
    "natural language processing involves understanding human language",
    "machine learning algorithms can predict text completions"
]
autocomplete.train(corpus)

# Save the trained model
autocomplete.save_model("models/my_model.pkl")
```

### Getting Autocomplete Suggestions

```python
# Load a pre-trained model
autocomplete = NGramAutocomplete()
autocomplete.load_model("models/my_model.pkl")

# Get suggestions
suggestions = autocomplete.get_suggestions("machine learning", top_k=5)
print(suggestions)
# Output: [('algorithms', 0.4), ('models', 0.3), ...]
```

### Generating Text

```python
# Generate text from a seed
generated_text = autocomplete.generate_text("the quick", max_length=20)
print(generated_text)
```

### Evaluating Model Performance

```python
# Calculate perplexity on test data
test_corpus = ["the quick brown fox", "machine learning is fascinating"]
perplexity = autocomplete.evaluate_perplexity(test_corpus)
print(f"Model perplexity: {perplexity}")
```

## Advanced Features

### Smoothing Methods

The system implements multiple smoothing techniques to handle the sparsity problem:

- **No Smoothing**: Basic maximum likelihood estimation with backoff
- **Laplace Smoothing**: Adds a small count to all possible words to avoid zero probabilities
- **Kneser-Ney Smoothing**: State-of-the-art method that considers how many different contexts a word appears in

### Parameters

- `n`: Size of N-grams (default: 3)
- `smoothing`: Smoothing method (options: "kneser_ney", "laplace", None)
- `discount`: Discount parameter for Kneser-Ney smoothing (default: 0.75)
- `top_k`: Number of suggestions to return (default: 5)
- 
## Web Application Features

- **Model Training**: Upload and train on custom text
- **Real-time Suggestions**: Get predictions as you type
- **Parameter Tuning**: Adjust N-gram size and smoothing techniques
- **Text Generation**: Create text from seed phrases
- **Model Comparison**: Evaluate different configurations

## Enhancing the N-gram Model

To improve the performance and capabilities of the N-gram system, consider these enhancements:

1. **Interpolated Smoothing**: Implement linear interpolation between different order N-grams for better probability estimates
   ```python
   # Combining bigram and trigram probabilities
   final_prob = lambda1 * unigram_prob + lambda2 * bigram_prob + lambda3 * trigram_prob
   ```

2. **Domain-Specific Corpora**: Train specialized models for different domains (medical, legal, technical) to improve suggestion relevance

3. **Spelling Correction**: Integrate edit distance algorithms (Levenshtein) to handle misspelled inputs
   ```python
   # Find closest match if input is not in vocabulary
   if input_word not in vocabulary:
       suggestions = get_closest_words(input_word, vocabulary, max_distance=2)
   ```

4. **Pruning Techniques**: Implement entropy-based or count-based pruning to reduce model size
   ```python
   # Remove n-grams with counts below threshold
   pruned_ngrams = {context: completions for context, completions in ngram_freq.items() 
                   if sum(completions.values()) >= min_count}
   ```

    ```
### Applications:
- **Search Engines**: Suggest relevant query completions based on user input.
- **Text Editors**: Predict next words or phrases for faster typing.
- **Coding Environments**: Provide autocompletion for code keywords or syntax.
- **Messaging Apps**: Generate typing suggestions for faster communication.

This code demonstrates how to efficiently build and use an n-gram model to provide real-time autocomplete predictions for a variety of applications.

--- 

## References

This project draws from the following resources:

1. Jurafsky, D., & Martin, J. H. (2023). [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf). Chapter 3: N-gram Language Models.
2. [Google Books Ngram Viewer](https://books.google.com/ngrams/info) - A tool that charts frequencies of words or phrases in Google's text corpora.
3. [Google Books Ngram Viewer - Wikipedia](https://en.wikipedia.org/wiki/Google_Books_Ngram_Viewer) - Background and history of the Google Books Ngram project.
   

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License.
