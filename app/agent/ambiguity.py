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
        movie_titles: list[str] | None = None,
    ) -> AmbiguityResult:
        """Check whether retrieved sources are ambiguous across movies.

        Args:
            sources:
                The list of retrieved source dicts (each must contain
                ``movie_title``).

            movie_title:
                Backward-compatible single explicit movie selection.
                When provided, ambiguity is always suppressed.

            movie_titles:
                Explicit multi-movie selection from the UI/caller.
                When one or more titles are provided, the user has
                already chosen which movie(s) to query, so cross-movie
                ambiguity is suppressed.
        """

        # ----------------------------------------------------------
        # Normalise: treat a single movie_title as a one-element list
        # so the rest of the logic is uniform.
        # ----------------------------------------------------------
        if movie_titles is None and movie_title:
            movie_titles = [movie_title]

        # If the caller made an explicit selection (any non-empty list),
        # ambiguity at the movie level is already resolved.
        if movie_titles:
            return AmbiguityResult(
                is_ambiguous=False,
                movie_titles=list(movie_titles),
            )

        if not sources:
            return AmbiguityResult(
                is_ambiguous=False,
                movie_titles=[],
            )

        # Preserve the order in which movies appear in retrieval.
        seen_titles: list[str] = []

        for source in sources:
            title = source.get("movie_title")

            if title and title not in seen_titles:
                seen_titles.append(title)

        # Multiple movies were retrieved with no explicit selection.
        if len(seen_titles) > 1:
            clarification = (
                "I found relevant information in multiple movies. "
                "Which movie are you referring to? "
                f"Possible matches: {', '.join(seen_titles[:5])}."
            )

            return AmbiguityResult(
                is_ambiguous=True,
                movie_titles=seen_titles,
                clarification=clarification,
            )

        return AmbiguityResult(
            is_ambiguous=False,
            movie_titles=seen_titles,
        )