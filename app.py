import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ngram_autocomplete import NGramAutocomplete

# Set page configuration with favicon
st.set_page_config(
    page_title="N-Gram Autocomplete",
    layout="wide",
    page_icon=":chart_with_upwards_trend:"
)

# Enhanced CSS for Professional Dark Theme with improved visibility
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Roboto+Mono&display=swap');
        
        /* Background & Base Styling */
        body, [class*="st-"], .block-container {
            background-color: #121212;
            color: #FFFFFF;  /* Changed from #E0E0E0 to white */
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main Header with 3D effect */
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            color: #1f77b4;
            margin: 2rem 0 1.5rem 0;
            text-shadow: 0px 4px 8px rgba(0,0,0,0.5);
            letter-spacing: 1px;
            background: linear-gradient(135deg, #1f77b4, #4dabf7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 0.5rem;
        }
        
        /* Section Headers with subtle 3D effect */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2ca02c;
            margin: 2rem 0 1.2rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #2ca02c;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: 0.5px;
        }
        
        /* Info Text - Changed color from grey to white */
        .info-text {
            font-size: 1.1rem;
            color: #FFFFFF;  /* Changed from #b0b0b0 to white */
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }
        
        /* Simplified the highlight section - removed the box effect */
        .highlight {
            padding: 1.5rem 0;
            margin: 1rem 0;
        }
        
        /* Enhanced Buttons with 3D effect */
        button {
            background: linear-gradient(to bottom, #1f77b4, #1565c0) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3), 
                        inset 0 1px 0 rgba(255,255,255,0.1) !important;
            transition: all 0.2s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            margin: 1rem 0 !important;
        }
        
        button:hover {
            background: linear-gradient(to bottom, #1565c0, #0d47a1) !important;
            box-shadow: 0 6px 10px rgba(0,0,0,0.4), 
                        inset 0 1px 0 rgba(255,255,255,0.1) !important;
            transform: translateY(-2px) !important;
        }
        
        button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            color: white;  /* Changed from #b0b0b0 to white */
            border: 1px solid #333;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        /* Form elements styling */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: #2d2d2d !important;
            border: 1px solid #444 !important;
            border-radius: 8px !important;
            color: white !important;  /* Changed from #e0e0e0 to white */
            padding: 12px !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
            font-family: 'Roboto Mono', monospace !important;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
            border-color: #1f77b4 !important;
            box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.3), 
                        inset 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        /* Slider styling */
        .stSlider [data-baseweb="slider"] {
            margin: 1.5rem 0 !important;
        }
        
        .stSlider [data-baseweb="thumb"] {
            background-color: #1f77b4 !important;
            border: 2px solid #fff !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
            height: 20px !important;
            width: 20px !important;
        }
        
        .stSlider [data-baseweb="track"] {
            background-color: #444 !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        
        .stSlider [data-baseweb="tick"] {
            background-color: #666 !important;
        }
        
        /* Data display */
        .stDataFrame {
            margin: 1.5rem 0 !important;
        }
        
        [data-testid="stTable"] {
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        /* Warning and success messages */
        .stAlert {
            background-color: #1e1e1e !important;
            border-radius: 8px !important;
            border-left-width: 4px !important;
            padding: 1rem !important;
            margin: 1.5rem 0 !important;
        }
        
        /* Results section - simplified */
        .result-text {
            background-color: #2d2d2d;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 1.5rem 0;
            font-family: 'Roboto Mono', monospace;
            line-height: 1.6;
            color: white;  /* Ensuring text is white */
        }
        
        /* Footer - simplified */
        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 2.5rem;
            font-size: 0.9rem;
            color: white;  /* Changed from #888 to white */
            border-top: 1px solid #333;
        }
        
        /* Customize Matplotlib plots for dark theme - simplified */
        .matplotlib-wrapper {
            background-color: transparent !important;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = NGramAutocomplete()

if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

if 'training_stats' not in st.session_state:
    st.session_state.training_stats = None

if 'suggestions_df' not in st.session_state:
    st.session_state.suggestions_df = None

if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""

# Set matplotlib style for dark theme
plt.style.use('dark_background')

# App header
st.markdown('<div class="main-header">N-Gram Autocomplete System</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">An advanced natural language processing tool for training and testing N-gram language models. Use this system to predict text, generate content, and evaluate language model performance.</div>', unsafe_allow_html=True)

# Create tabs for different functionality
tab1, tab2, tab3, tab4 = st.tabs(["Train Model", "Get Suggestions", "Generate Text", "Evaluate Model"])

# Tab 1: Train Model
with tab1:
    st.markdown('<div class="section-header">Train a New Model</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input for training data
            st.markdown("### Training Data")
            training_option = st.radio(
                "Choose input method:",
                ("Upload a text file", "Enter text directly"),
                key="training_input_method"
            )
            
            training_text = ""
            
            if training_option == "Upload a text file":
                uploaded_file = st.file_uploader("Upload a text file", type=["txt"], key="training_file_uploader")
                if uploaded_file is not None:
                    try:
                        training_text = uploaded_file.getvalue().decode("utf-8")
                        st.success(f"File loaded: {len(training_text)} characters, approximately {len(training_text.split())} words")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                training_text = st.text_area(
                    "Enter training text:",
                    height=200,
                    placeholder="Enter or paste your training text here...",
                    key="training_text_area"
                )
                if training_text:
                    st.info(f"Text length: {len(training_text)} characters, approximately {len(training_text.split())} words")
        
        with col2:
            # Model parameters
            st.markdown("### Model Parameters")
            n_value = st.slider("N-gram Size", min_value=1, max_value=5, value=3, 
                              help="Size of the N-gram context (higher values capture more context but require more data)",
                              key="n_gram_slider")
            
            smoothing_method = st.selectbox(
                "Smoothing Method",
                options=["none", "laplace", "kneser_ney"],
                index=2,
                help="Smoothing technique to handle unseen N-grams",
                key="smoothing_method"
            )
            
            discount_factor = st.slider("Discount Factor", min_value=0.1, max_value=0.9, value=0.75, step=0.05,
                                      help="Discount factor for Kneser-Ney smoothing (only used with Kneser-Ney)",
                                      key="discount_factor")
            
            model_filename = st.text_input("Model filename", "autocomplete_model.pkl", key="model_filename")
        
        # Train button
        train_button = st.button("Train Model", disabled=not training_text, key="train_button")
        if train_button:
            with st.spinner("Training model..."):
                # Initialize model with selected parameters
                smoothing = None if smoothing_method == "none" else smoothing_method
                st.session_state.model = NGramAutocomplete(n=n_value, smoothing=smoothing, discount=discount_factor)
                
                # Train the model
                st.session_state.model.train(training_text)
                
                # Save model
                if model_filename:
                    try:
                        success = st.session_state.model.save_model(model_filename)
                        if success:
                            st.success(f"Model saved to {model_filename}")
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
                
                # Update session state
                st.session_state.is_trained = True
                
                # Collect training statistics
                st.session_state.training_stats = {
                    "Vocabulary Size": st.session_state.model.vocab_size,
                    "Total Words": st.session_state.model.total_words,
                    "N-gram Size": st.session_state.model.n,
                    "Smoothing Method": st.session_state.model.smoothing or "None",
                    "Top Words": dict(sorted(st.session_state.model.word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                }
                
                st.success("Model trained successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load existing model
    st.markdown('<div class="section-header">Or Load Existing Model</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        load_filename = st.text_input("Path to existing model file", key="load_model_path")
        
        load_button = st.button("Load Model", disabled=not load_filename, key="load_model_button")
        if load_button:
            with st.spinner("Loading model..."):
                try:
                    success = st.session_state.model.load_model(load_filename)
                    if success:
                        st.session_state.is_trained = True
                        
                        # Update training statistics
                        st.session_state.training_stats = {
                            "Vocabulary Size": st.session_state.model.vocab_size,
                            "Total Words": st.session_state.model.total_words,
                            "N-gram Size": st.session_state.model.n,
                            "Smoothing Method": st.session_state.model.smoothing or "None",
                            "Top Words": dict(sorted(st.session_state.model.word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                        }
                        
                        st.success(f"Model loaded from {load_filename}")
                    else:
                        st.error("Failed to load model")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display model statistics
    if st.session_state.is_trained and st.session_state.training_stats:
        st.markdown('<div class="section-header">Model Statistics</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### General Statistics")
                stats = st.session_state.training_stats
                st.markdown(f"- **Vocabulary Size:** {stats['Vocabulary Size']} unique words")
                st.markdown(f"- **Total Words:** {stats['Total Words']} words")
                st.markdown(f"- **N-gram Size:** {stats['N-gram Size']}")
                st.markdown(f"- **Smoothing Method:** {stats['Smoothing Method']}")
            
            with col2:
                st.markdown("#### Top 10 Most Frequent Words")
                
                # Plot top words
                top_words = stats["Top Words"]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(top_words.keys(), top_words.values(), color="#1f77b4")
                ax.set_xlabel("Word")
                ax.set_ylabel("Frequency")
                ax.set_title("Top 10 Words")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Get Suggestions
with tab2:
    st.markdown('<div class="section-header">Get Autocomplete Suggestions</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        if not st.session_state.is_trained:
            st.warning("Please train or load a model first!")
        else:
            # Input for text
            input_text = st.text_input("Enter text for suggestions:", placeholder="Type here...", key="suggestion_input")
            top_k = st.slider("Number of suggestions:", min_value=1, max_value=20, value=5, key="top_k_slider")
            
            suggestions_button = st.button("Get Suggestions", disabled=not input_text, key="get_suggestions_button")
            if suggestions_button:
                with st.spinner("Generating suggestions..."):
                    suggestions = st.session_state.model.get_suggestions(input_text, top_k=top_k)
                    
                    if suggestions:
                        # Convert to DataFrame for better display
                        suggestions_data = {
                            "Word": [word for word, prob in suggestions],
                            "Probability": [prob for word, prob in suggestions]
                        }
                        st.session_state.suggestions_df = pd.DataFrame(suggestions_data)
                    else:
                        st.session_state.suggestions_df = None
                        st.info("No suggestions found for the given input.")
            
            # Display suggestions
            if st.session_state.suggestions_df is not None:
                st.markdown(f"### Suggestions for: '{input_text}'")
                
                # Display as table
                st.dataframe(st.session_state.suggestions_df, use_container_width=True)
                
                # Visualize probabilities
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(st.session_state.suggestions_df["Word"], st.session_state.suggestions_df["Probability"], color="#2ca02c")
                ax.set_xlabel("Word")
                ax.set_ylabel("Probability")
                ax.set_title("Suggestion Probabilities")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Generate Text
with tab3:
    st.markdown('<div class="section-header">Generate Text</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        if not st.session_state.is_trained:
            st.warning("Please train or load a model first!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                seed_text = st.text_input("Seed text (optional):", placeholder="Start with...", key="seed_text_input")
            
            with col2:
                max_length = st.slider("Maximum length (words):", min_value=5, max_value=100, value=20, key="max_length_slider")
            
            generate_button = st.button("Generate Text", key="generate_text_button")
            if generate_button:
                with st.spinner("Generating text..."):
                    st.session_state.generated_text = st.session_state.model.generate_text(
                        seed_text=seed_text,
                        max_length=max_length
                    )
            
            if st.session_state.generated_text:
                st.markdown("### Generated Text:")
                st.markdown(f"**Seed:** '{seed_text or '(empty)'}'")
                st.markdown(f'<div class="result-text">{st.session_state.generated_text}</div>', unsafe_allow_html=True)
                
                # Copy button
                copy_button = st.button("Copy to clipboard", key="copy_text_button")
                if copy_button:
                    st.code(st.session_state.generated_text)
                    st.info("Use Ctrl+C to copy the text above")
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Evaluate Model
with tab4:
    st.markdown('<div class="section-header">Evaluate Model</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        if not st.session_state.is_trained:
            st.warning("Please train or load a model first!")
        else:
            st.markdown("### Model Evaluation using Perplexity")
            st.markdown("""
            Perplexity measures how well the model predicts a sample. Lower perplexity indicates better performance.
            
            Enter test text to evaluate the model's perplexity:
            """)
            
            # Text input for test data
            test_option = st.radio(
                "Choose input method:",
                ("Upload a text file", "Enter text directly"),
                key="test_input_method"
            )
            
            test_text = ""
            
            if test_option == "Upload a text file":
                test_file = st.file_uploader("Upload a test text file", type=["txt"], key="test_file_uploader")
                if test_file is not None:
                    try:
                        test_text = test_file.getvalue().decode("utf-8")
                        st.success(f"Test file loaded: {len(test_text)} characters")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                test_text = st.text_area(
                    "Enter test text:",
                    height=150,
                    placeholder="Enter or paste your test text here...",
                    key="test_text_area"
                )
            
            perplexity_button = st.button("Calculate Perplexity", disabled=not test_text, key="calculate_perplexity_button")
            if perplexity_button:
                with st.spinner("Calculating perplexity..."):
                    try:
                        # Calculate perplexity directly on the test text
                        perplexity = st.session_state.model.evaluate_perplexity(test_text)
                        
                        st.markdown("### Perplexity Results")
                        
                        # Display perplexity with interpretation
                        st.markdown(f'<div class="result-text"><strong>Perplexity Score:</strong> {perplexity:.4f}</div>', unsafe_allow_html=True)
                        
                        # Interpretation guide
                        if perplexity < 10:
                            st.success("Excellent! The model has very good predictive power for this text.")
                        elif perplexity < 50:
                            st.info("Good. The model has reasonable predictive power for this text.")
                        elif perplexity < 100:
                            st.warning("Fair. The model has some difficulties predicting this text.")
                        else:
                            st.error("Poor. The model struggles to predict this text accurately.")
                        
                        st.markdown("*Lower perplexity indicates better model performance.*")
                    except Exception as e:
                        st.error(f"Error calculating perplexity: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# App footer
st.markdown('<div class="footer"> N-Gram Autocomplete System </div>', unsafe_allow_html=True)