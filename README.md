# ContentWeaver AI 📰

ContentWeaver AI is an intelligent newsletter generator that creates personalized digests based on your interests. It uses AI to curate, summarize, and provide commentary on relevant articles from various sources.

## Features

- 🔍 Personalized content curation based on your keywords
- 📝 AI-powered article summarization
- 💭 Smart commentary generation with customizable tone
- 🔄 Real-time RSS feed integration
- 🎯 Semantic search using vector embeddings
- 📱 Clean, user-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- Hugging Face API token (for LLM access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/b-isry/ContentWeaverAI.git
cd ContentWeaverAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Hugging Face token:
```
HF_Token=your_huggingface_token
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app/main.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. In the sidebar:
   - Enter your topics/keywords (comma-separated)
   - Select your preferred newsletter tone
   - Click "Craft" to generate your personalized newsletter

## How It Works

1. **Content Collection**: Fetches articles from configured RSS feeds
2. **Semantic Search**: Uses sentence transformers to find relevant articles based on your keywords
3. **AI Processing**: 
   - Generates concise summaries of selected articles
   - Creates engaging commentary with your preferred tone
4. **Newsletter Generation**: Compiles everything into a well-formatted markdown newsletter

## Dependencies

- Streamlit: Web interface
- Transformers: LLM integration
- Sentence-Transformers: Text embeddings
- ChromaDB: Vector database
- BeautifulSoup4: Web scraping
- Feedparser: RSS feed parsing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 