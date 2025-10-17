document.addEventListener('DOMContentLoaded', () => {
  const searchBtn = document.getElementById('search-btn');
  const resultsContainer = document.getElementById('results');

  function clearResults() {
    resultsContainer.innerHTML = '';
  }

  function renderMovies(movies, heading) {
    if (movies.length === 0) {
      const noResult = document.createElement('p');
      noResult.textContent = `No ${heading.toLowerCase()} found.`;
      resultsContainer.appendChild(noResult);
      return;
    }
    const sectionTitle = document.createElement('h2');
    sectionTitle.textContent = heading;
    sectionTitle.classList.add('results-heading');
    resultsContainer.appendChild(sectionTitle);
    const grid = document.createElement('div');
    grid.classList.add('results-section');
    movies.forEach(movie => {
      const card = document.createElement('div');
      card.classList.add('result-card');
      const title = document.createElement('h3');
      title.textContent = movie.title;
      const genres = document.createElement('p');
      genres.classList.add('genres');
      genres.textContent = movie.genres.join(', ');
      const overview = document.createElement('p');
      overview.classList.add('overview');
      // Show first 200 characters of the overview
      overview.textContent = movie.overview.slice(0, 200) + (movie.overview.length > 200 ? '…' : '');
      const score = document.createElement('p');
      score.classList.add('score');
      // If cosine_score exists, show it, else hide
      if (movie.cosine_score !== undefined) {
        score.textContent = `Score: ${movie.cosine_score.toFixed(3)}`;
      } else {
        score.textContent = '';
      }
      card.appendChild(title);
      card.appendChild(genres);
      card.appendChild(overview);
      card.appendChild(score);
      grid.appendChild(card);
    });
    resultsContainer.appendChild(grid);
  }

  searchBtn.addEventListener('click', async () => {
    clearResults();
    const queryInput = document.getElementById('query-input').value.trim();
    if (!queryInput) {
      alert('Please enter a description or movie title.');
      return;
    }
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const topN = parseInt(document.getElementById('top-n').value, 10);
    const endpoint = `http://127.0.0.1:8000${mode === 'title' ? '/recommend/title' : '/recommend/description'}`;
    const payload = mode === 'title' ? { title: queryInput, top_n: topN } : { query: queryInput, top_n: topN };
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const errorData = await response.json();
        alert(errorData.detail || 'Error fetching recommendations.');
        return;
      }
      const data = await response.json();
      const localRecs = data.recommendations?.local_retrieval || [];
      const mindsRecs = data.recommendations?.mindsdb || [];
      renderMovies(localRecs, 'Recommended Movies');
      // Optionally display MindsDB results if available
      if (mindsRecs.length > 0) {
        renderMovies(mindsRecs, 'MindsDB Suggestions');
      }
    } catch (err) {
      console.error(err);
      alert('An unexpected error occurred.');
    }
  });
});