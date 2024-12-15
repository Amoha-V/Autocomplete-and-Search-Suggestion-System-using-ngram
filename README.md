## Autocomplete System Using N-Grams

This code demonstrates the implementation of an **Autocomplete System** based on an n-gram model. It predicts word completions and suggestions using trained n-grams from a given text corpus.

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

### Applications:
- **Search Engines**: Suggest relevant query completions based on user input.
- **Text Editors**: Predict next words or phrases for faster typing.
- **Coding Environments**: Provide autocompletion for code keywords or syntax.
- **Messaging Apps**: Generate typing suggestions for faster communication.

This code demonstrates how to efficiently build and use an n-gram model to provide real-time autocomplete predictions for a variety of applications.

--- 
### License
This project is proprietary. All rights reserved. No part of this project may be reproduced, distributed, or transmitted in any form or by any means.
