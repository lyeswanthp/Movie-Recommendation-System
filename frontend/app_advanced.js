// Advanced AI Movie Recommender Frontend
// Connects to the FastAPI backend with AI agents

const API_BASE_URL = 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');
    const queryInput = document.getElementById('query-input');
    const resultsContainer = document.getElementById('results');
    const statusText = document.getElementById('status-text');
    const modelInfo = document.getElementById('model-info');

    // Check system status on load
    checkSystemStatus();

    // Add enter key support
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchBtn.click();
        }
    });

    // Main search handler
    searchBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();

        if (!query) {
            showError('Please enter a movie description or title.');
            return;
        }

        const mode = document.querySelector('input[name="mode"]:checked').value;
        const topN = parseInt(document.getElementById('top-n').value, 10);

        await performSearch(query, mode, topN);
    });

    async function checkSystemStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            if (!response.ok) throw new Error('Backend not available');

            const data = await response.json();

            if (data.ollama_available) {
                statusText.textContent = `🤖 AI Mode - ${data.vector_store_count} movies indexed`;
                const models = data.ollama_models.join(', ');
                modelInfo.textContent = `Models: ${models}`;
            } else {
                statusText.textContent = `⚡ Vector Search Mode - ${data.vector_store_count} movies indexed`;
                modelInfo.textContent = 'AI unavailable';
                showInfo('🔧 Ollama not detected. Using vector search mode. For AI-powered explanations, install Ollama and run: ollama pull llama3.2');
            }
        } catch (error) {
            statusText.textContent = '❌ Backend offline';
            modelInfo.textContent = 'Not connected';
            showError('Cannot connect to the backend server. Please ensure it is running on port 8000.');
        }
    }

    async function performSearch(query, mode, topN) {
        clearResults();
        showLoading();

        try {
            const payload = {
                query: query,
                mode: mode,
                top_n: topN
            };

            const response = await fetch(`${API_BASE_URL}/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error fetching recommendations');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            clearResults();
            showError(`Error: ${error.message}`);
        }
    }

    function clearResults() {
        resultsContainer.innerHTML = '';
    }

    function showLoading() {
        resultsContainer.innerHTML = '<div class="loading">Searching for perfect movies</div>';
    }

    function showError(message) {
        resultsContainer.innerHTML = `<div class="error">❌ ${message}</div>`;
    }

    function showInfo(message) {
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = `
            background: #dbeafe;
            border: 2px solid #3b82f6;
            color: #1e40af;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        `;
        infoDiv.textContent = message;
        resultsContainer.insertBefore(infoDiv, resultsContainer.firstChild);
    }

    function displayResults(data) {
        clearResults();

        const mode = data.mode || 'unknown';
        const recommendations = data.recommendations || {};
        const summary = recommendations.summary || 'Here are your recommendations';
        const movies = recommendations.movies || [];

        if (movies.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results">😔 No movies found matching your criteria. Try a different search!</div>';
            return;
        }

        // Summary box
        const summaryBox = document.createElement('div');
        summaryBox.className = 'summary-box';
        summaryBox.innerHTML = `
            <h3>🎯 ${summary}
                <span class="mode-badge">${mode === 'ai_powered' ? '🤖 AI Powered' : '⚡ Vector Search'}</span>
            </h3>
            <p>Found ${movies.length} amazing recommendations for you!</p>
        `;
        resultsContainer.appendChild(summaryBox);

        // Results grid
        const grid = document.createElement('div');
        grid.className = 'results-grid';

        movies.forEach((movie, index) => {
            const card = createMovieCard(movie, index + 1);
            grid.appendChild(card);
        });

        resultsContainer.appendChild(grid);
    }

    function createMovieCard(movie, rank) {
        const card = document.createElement('div');
        card.className = 'movie-card';

        // Title
        const title = document.createElement('div');
        title.className = 'movie-title';
        title.textContent = `${rank}. ${movie.title}`;

        // Genres
        const genresContainer = document.createElement('div');
        genresContainer.className = 'movie-genres';

        const genres = movie.genres || [];
        genres.forEach(genre => {
            if (genre) {
                const tag = document.createElement('span');
                tag.className = 'genre-tag';
                tag.textContent = genre;
                genresContainer.appendChild(tag);
            }
        });

        // Overview
        const overview = document.createElement('div');
        overview.className = 'movie-overview';
        const overviewText = movie.overview || 'No description available.';
        overview.textContent = overviewText.length > 200
            ? overviewText.slice(0, 200) + '...'
            : overviewText;

        // Score
        const scoreContainer = document.createElement('div');
        scoreContainer.className = 'movie-score';

        if (movie.similarity_score !== undefined) {
            const scorePercentage = (movie.similarity_score * 100).toFixed(0);
            scoreContainer.innerHTML = `
                <span class="score-badge">
                    ${scorePercentage}% Match
                </span>
            `;
        }

        // AI Explanation (if available)
        let aiExplanation = null;
        if (movie.ai_explanation) {
            aiExplanation = document.createElement('div');
            aiExplanation.className = 'ai-explanation';
            aiExplanation.innerHTML = `<strong>🤖 AI Insight:</strong> ${movie.ai_explanation}`;
        }

        // Assemble card
        card.appendChild(title);
        card.appendChild(genresContainer);
        card.appendChild(overview);
        card.appendChild(scoreContainer);
        if (aiExplanation) {
            card.appendChild(aiExplanation);
        }

        return card;
    }
});
