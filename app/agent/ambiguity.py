from dataclasses import dataclass


@dataclass
class AmbiguityResult:
    is_ambiguous: bool
    movie_titles: list[str]
    clarification: str | None = None


class AmbiguityHandler:
    def check(
        self,
        sources: list[dict],
        movie_title: str | None = None,
    ) -> AmbiguityResult:

        # If the user explicitly selected a movie,
        # there is no movie-level ambiguity.
        if movie_title:
            return AmbiguityResult(
                is_ambiguous=False,
                movie_titles=[movie_title],
            )

        if not sources:
            return AmbiguityResult(
                is_ambiguous=False,
                movie_titles=[],
            )

        # Preserve the order in which movies appear in retrieval.
        movie_titles = []

        for source in sources:
            title = source.get("movie_title")

            if title and title not in movie_titles:
                movie_titles.append(title)

        # Multiple movies were retrieved.
        if len(movie_titles) > 1:
            clarification = (
                "I found relevant information in multiple movies. "
                "Which movie are you referring to? "
                f"Possible matches: {', '.join(movie_titles[:5])}."
            )

            return AmbiguityResult(
                is_ambiguous=True,
                movie_titles=movie_titles,
                clarification=clarification,
            )

        return AmbiguityResult(
            is_ambiguous=False,
            movie_titles=movie_titles,
        )