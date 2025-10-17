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

### Step 2: Choose Your Version

#### Option A: Run the Basic Version

**Windows:**
```cmd
cd backend
python main.py
```

**Linux:**
```bash
cd backend
python3 main.py
```

The server will start at: http://127.0.0.1:8000

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

## Troubleshooting

### "Python not found" error:
- Make sure Python is installed and added to your PATH
- Try using `python3` instead of `python` (especially on Linux/Mac)

### Port already in use:
- Another app is using port 8000
- Either close that app or change the port in the code (look for `port=8000` in main.py)

### "Module not found" error:
- Make sure you activated the virtual environment
- Run the pip install command again
- Check that you're in the right directory

### Advanced version doesn't give AI explanations:
- This is normal if Ollama isn't installed or running
- The app will work fine with just vector search
- To enable AI features, install Ollama and pull a model (see Step 2, Option B)

### Movies not loading:
- Check that the `backend/data/movielens_metadata.csv` file exists
- The first time you run the advanced version, it needs to build the vector database (takes a few minutes)

### Frontend can't connect to backend:
- Make sure the backend server is running (you should see logs in the terminal)
- Check that you're using the correct address (http://127.0.0.1:8000)
- Look for any error messages in the terminal where the server is running

## Technical Details (For the Curious)

### Technologies Used:
- **FastAPI**: Modern Python web framework (handles the API)
- **pandas**: Data manipulation (loads and processes movie data)
- **scikit-learn**: TF-IDF vectorization (basic version search)
- **ChromaDB**: Vector database (advanced version storage)
- **Sentence Transformers**: Creates semantic embeddings (advanced version)
- **LangGraph**: Multi-agent workflow system (coordinates the AI)
- **Ollama**: Local LLM inference (AI explanations, optional)
- **Vanilla JavaScript**: Frontend interactivity (no frameworks needed)

### How It Works:

**Basic Version:**
1. Loads movie data into memory
2. Creates TF-IDF vectors from movie descriptions
3. When you search, it converts your query to a vector
4. Finds movies with vectors most similar to your query
5. Returns the top matches

**Advanced Version:**
1. Loads movies and creates semantic embeddings using Sentence Transformers
2. Stores embeddings in ChromaDB for fast retrieval
3. When you search, creates an embedding of your query
4. Finds nearest neighbors in vector space (semantic similarity)
5. If Ollama is available, uses AI to explain why each movie matches
6. Returns recommendations with AI-generated explanations

## Dataset

The app uses the MovieLens dataset, which contains thousands of movies with descriptions, genres, and metadata. This is real data from actual movie databases, so you're getting quality recommendations.

## Future Ideas

Some things that could make this even better:
- User accounts to save favorite movies
- Rating system to improve recommendations over time
- Integration with streaming service availability
- Movie trailers and poster images
- Social features to share recommendations with friends
- Mobile app version

## Need Help?

If something's not working:
1. Check the troubleshooting section above
2. Make sure all the setup steps were completed
3. Look at the terminal where the server is running for error messages
4. Try restarting the server

## License

This is a learning project built for educational purposes. The MovieLens data is used under their license terms.

---

Made with care to help you discover your next favorite movie. Happy watching!
