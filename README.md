# Movie Recommender App

A smart movie recommendation system that helps you discover films based on what you're looking for. Think of it as your personal movie advisor that understands what you want to watch, even when you describe it in your own words.

## What This App Does

This app gives you two ways to find great movies:

1. **Describe what you want**: Tell it something like "I want a funny movie about time travel" or "looking for dark psychological thriller" and it will find movies that match
2. **Find similar movies**: Give it a movie title you like, and it'll recommend similar movies you might enjoy

The app uses some clever technology under the hood (vector search, AI agents, and machine learning) to understand your requests and find the best matches.

## Two Versions Available

### Basic Version (Simple and Fast)
- Uses TF-IDF search to match movies
- Works offline, no internet needed
- Perfect for quick recommendations
- Lightweight and straightforward

### Advanced Version (AI-Powered)
- Uses semantic search with ChromaDB (understands meaning, not just keywords)
- Optional AI explanations using Ollama (local LLM - no API keys needed!)
- Smarter recommendations with context understanding
- Conversational chat interface

## What You Need

### For Windows Users:
- Python 3.8 or newer (download from python.org if you don't have it)
- A few Python libraries (we'll install these together)
- For the advanced version with AI features: Ollama installed locally (optional)

### For Linux Users:
- Python 3.8 or newer (probably already installed)
- pip package manager (usually comes with Python)
- For the advanced version with AI features: Ollama installed locally (optional)

## Getting Started

### Step 1: Setting Up (Windows)

1. Open Command Prompt (press Windows key, type "cmd", press Enter)

2. Navigate to the project folder:
```cmd
cd Downloads\movie_recommender_app
```

3. Create a virtual environment (this keeps the project tidy):
```cmd
python -m venv venv
```

4. Activate the virtual environment:
```cmd
venv\Scripts\activate
```

5. Install required packages:
```cmd
pip install -r backend\requirements.txt
```

### Step 1: Setting Up (Linux)

1. Open Terminal (Ctrl+Alt+T)

2. Navigate to the project folder:
```bash
cd ~/Downloads/movie_recommender_app
```

3. Create a virtual environment:
```bash
python3 -m venv venv
```

4. Activate the virtual environment:
```bash
source venv/bin/activate
```

5. Install required packages:
```bash
pip install -r backend/requirements.txt
```

#### Option B: Run the Advanced Version (with AI)

First, if you want AI-powered explanations, install Ollama:
- Windows/Linux: Visit https://ollama.ai and download for your system
- After installing, open a terminal and run: `ollama pull llama3.2`

Then start the advanced server:

**Windows:**
```cmd
cd backend
python main_advanced.py
```

**Linux:**
```bash
cd backend
python3 main_advanced.py
```

The advanced server will start at: http://127.0.0.1:8000

Note: The advanced version works even without Ollama, but you won't get the AI explanations - it'll just use vector search.

### Step 3: Open the Web Interface

1. Open your web browser (Chrome, Firefox, Edge, etc.)

2. For basic version, open: `frontend/index.html`

3. For advanced version, open: `frontend/index_advanced.html`

You can just drag and drop these files into your browser, or navigate to them through File > Open.

### Step 4: Start Discovering Movies!

Once you're in:
- Type what you're looking for in the search box
- Choose "Description" if you're describing what you want
- Choose "Movie Title" if you want movies similar to one you already like
- Pick how many results you want (3, 5, or 10)
- Hit Search and explore your recommendations!

## Project Structure

Here's what's in the project (the important stuff):

```
movie_recommender_app/
├── backend/                    # Server-side code
│   ├── main.py                # Basic version server
│   ├── main_advanced.py       # Advanced AI version server
│   ├── data_loader.py         # Loads movie data (basic version)
│   ├── vector_store.py        # Smart semantic search (advanced version)
│   ├── ai_agent.py            # AI explanation generator (advanced version)
│   ├── graph.py               # Search workflow (basic version)
│   ├── mindsdb_client.py      # Optional MindsDB integration
│   ├── requirements.txt       # List of needed Python packages
│   └── data/                  # Movie database
│       └── movielens_metadata.csv
│
├── frontend/                   # Web interface
│   ├── index.html             # Basic version webpage
│   ├── index_advanced.html    # Advanced version webpage
│   ├── app.js                 # Basic version JavaScript
│   ├── app_advanced.js        # Advanced version JavaScript
│   └── styles.css             # Makes it look nice
│
└── venv/                       # Python virtual environment (created during setup)
```

## Features Breakdown

### What the Basic Version Offers:
- Fast keyword-based movie search
- Search by movie description or title
- Returns top matching movies with similarity scores
- Shows movie title, genres, and description
- Works completely offline
- Lightweight - runs on any machine

### What the Advanced Version Adds:
- Semantic understanding (knows "scary" and "horror" are related)
- AI-generated explanations for why each movie is recommended
- Better handling of complex queries
- Conversational memory (if you use the chat endpoint)
- Faster similarity search with ChromaDB
- Works without Ollama (falls back to vector search only)

## Dataset

The app uses the MovieLens dataset, which contains thousands of movies with descriptions, genres, and metadata. This is real data from actual movie databases, so you're getting quality recommendations.

---

Made with care to help you discover your next favorite movie. Happy watching!
