document.addEventListener("DOMContentLoaded", function () {
    const genreSearchButton = document.getElementById("genre-search-button");
    const genreInput = document.getElementById("genre-input");
    const genreRecommendationsDiv = document.getElementById("genre-recommendations");

    genreSearchButton.addEventListener("click", function () {
        const selectedGenre = genreInput.value.trim();

        if (selectedGenre) {
            fetch(`/recommend_by_genre?genre=${encodeURIComponent(selectedGenre)}`)
                .then(response => response.json())
                .then(data => {
                    genreRecommendationsDiv.innerHTML = "";
                    if (data.titles && data.posters) {
                        data.titles.forEach((title, index) => {
                            const card = document.createElement("div");
                            card.className = "recommendation-card";

                            const poster = document.createElement("img");
                            poster.className = "movie-poster";
                            poster.src = data.posters[index];
                            poster.alt = `${title} poster`;
                            card.appendChild(poster);

                            const movieTitle = document.createElement("h5");
                            movieTitle.className = "movie-title";
                            movieTitle.textContent = title;
                            card.appendChild(movieTitle);

                            genreRecommendationsDiv.appendChild(card);
                        });
                    } else {
                        genreRecommendationsDiv.innerHTML = "<p>No movies found for this genre.</p>";
                    }
                })
                .catch(error => {
                    console.error("Error fetching genre recommendations:", error);
                    genreRecommendationsDiv.innerHTML = "<p>Error fetching recommendations. Please try again.</p>";
                });
        }
    });
});
