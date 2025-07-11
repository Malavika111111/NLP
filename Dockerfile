# Use base Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora (including wordnet) — fix for WordNet error
RUN python -m nltk.downloader punkt stopwords wordnet && \
    mkdir -p /root/nltk_data && \
    python -c "import nltk; nltk.download('wordnet',
                                          download_dir='/root/nltk_data')" && \
    python -c "import nltk; nltk.download('stopwords',
                                          download_dir='/root/nltk_data')" && \
    python -c "import nltk; nltk.download('punkt', 
                                          download_dir='/root/nltk_data')"

# Set environment so NLTK knows where to look
ENV NLTK_DATA=/root/nltk_data

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", 
     "--server.enableCORS=false"]
