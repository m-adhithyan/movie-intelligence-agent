from app.agent.agent import MovieAgent


def main():
    agent = MovieAgent()

    request = "What does the artist say about art?"

    result = agent.run(request)

    print("\nUSER:")
    print(request)

    print("\nAGENT INTENT:")
    print(result["intent"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"- {source.get('movie_title')} | "
            f"{source.get('start_time')} --> "
            f"{source.get('end_time')}"
        )


if __name__ == "__main__":
    main()