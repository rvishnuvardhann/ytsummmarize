import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from nltk.corpus import stopwords
import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, PegasusForConditionalGeneration, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define functions to perform summarization and evaluation tasks


@st.cache_data(show_spinner=False)
def summarize(text):
    summarization = pipeline('summarization')
    summarized_text = summarization(text)
    return summarized_text[0]['summary_text']


@st.cache_data(show_spinner=False)
def pegasus_summarize(text):
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(
        model_name).to(device)
    batch = tokenizer(text, truncation=True, padding='longest',
                      return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    tgt_text = tgt_text.replace('<n>', '\n')
    return tgt_text


def evaluate_cosine_similarity(t1, t2):
    vectorizer = CountVectorizer()
    vectorized_text = vectorizer.fit_transform([t1, t2])
    similarity_score = cosine_similarity(vectorized_text)[0][1]
    return similarity_score


def evaluate_f1_score(t1, t2):
    ref_tokens = nltk.word_tokenize(t2.lower())
    gen_tokens = nltk.word_tokenize(t1.lower())
    tp = len(set(ref_tokens) & set(gen_tokens))
    fp = len(gen_tokens) - tp
    fn = len(ref_tokens) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return f1_score


def evaluate_rouge_score(t1, t2):
    rouge = Rouge()
    scores = rouge.get_scores(t1, t2)
    return scores


def evaluate_polarity_score(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores['compound']
    return polarity

# Define Streamlit app


def app():
    st.title('YouTube Video Transcription Summarization')

    # Get input from user
    video_url = st.text_input('Enter YouTube video link:')
    if not video_url:
        st.warning('Please enter a valid YouTube video link')
        return

    try:
        with st.spinner("Hang tight, we're working on it!"):
            # Get transcript from YouTube video
            video_id = video_url.split("=")[1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            result = ""
            for i in transcript:
                result += ' ' + i['text']
                result = ' '.join(result.split()[:512])

            # Summarize the transcript
            summarized_text = summarize(result)

            # Use Pegasus to summarize the transcript
            pegasus_summarized_text = pegasus_summarize(result)

            # Format the translated text
            sentences = pegasus_summarized_text .split('. ')
            bullet_points = [f'* {s}' for s in sentences]

            # Evaluate similarity between pegasus_summarized_text and summarized transcript
            similarity_score = evaluate_cosine_similarity(
                pegasus_summarized_text, summarized_text)

            # Evaluate F1 score between pegasus_summarized_text and summarized transcript
            f1_score = evaluate_f1_score(
                pegasus_summarized_text, summarized_text)

            # Evaluate Rouge score between pegasus_summarized_text and summarized transcript
            rouge_score = evaluate_rouge_score(
                pegasus_summarized_text, summarized_text)

            # Evaluate polarity score of pegasus_summarized_text
            polarity_score = evaluate_polarity_score(pegasus_summarized_text)

        # Display results to user
        st.header('Summary of Transcript')
        st.write('\n'.join(bullet_points))

        st.write('')
        st.write('')

        # Define stop words
        stop_words = set(stopwords.words('english'))
        # Add custom stop words
        custom_stop_words = ['video', 'watch',
                             'transcript', 'like', 'share', 'subscribe']
        stop_words.update(custom_stop_words)

        wc = WordCloud(width=800, height=400, background_color="white",
                       stopwords=stop_words, colormap="Dark2").generate(summarized_text)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.header('Evaluation Metrics')
        st.write('Cosine Similarity Score:', similarity_score)
        st.write('F1 Score:', f1_score)
        st.write('Rouge Score:', rouge_score[0]['rouge-1']['f'])
        st.write('Polarity Score:', polarity_score)

        # Calculate the compression ratio
        compressed_length = len(pegasus_summarized_text)
        original_length = len(result)
        compression_ratio = round(
            (compressed_length / original_length) * 100, 2)

        st.write(f'**Compression ratio:** {compression_ratio:.2f}%')

    except Exception as e:
        st.warning('Error occurred: ' + str(e))


# Run Streamlit app
app()
