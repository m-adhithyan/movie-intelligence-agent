from app.agent.ambiguity import AmbiguityHandler


handler = AmbiguityHandler()


# ---------------------------------------------------------
# TEST 1: One movie
# ---------------------------------------------------------

sources_one_movie = [
    {
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "movie_title": "A Bucket Of Blood 1959",
    },
]

result = handler.check(sources_one_movie)

print("\nTEST 1: One movie")
print(result)


# ---------------------------------------------------------
# TEST 2: Multiple movies
# ---------------------------------------------------------

sources_multiple_movies = [
    {
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "movie_title": "A Bucket Of Blood 1959",
    },
    {
        "movie_title": "Another Movie 1960",
    },
]

result = handler.check(sources_multiple_movies)

print("\nTEST 2: Multiple movies")
print(result)


# ---------------------------------------------------------
# TEST 3: Explicit movie supplied
# ---------------------------------------------------------

result = handler.check(
    sources_multiple_movies,
    movie_title="A Bucket Of Blood 1959",
)

print("\nTEST 3: Explicit movie")
print(result)


# ---------------------------------------------------------
# TEST 4: No sources
# ---------------------------------------------------------

result = handler.check([])

print("\nTEST 4: No sources")
print(result)