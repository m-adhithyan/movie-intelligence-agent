from app.agent.router import AgentRouter


router = AgentRouter()


test_cases = [
    "What happens when Walter meets the artist?",

    "Email me a breakdown of the scene where Walter creates "
    "the sculpture to alice@example.com",

    "Email me the scene",

    "Send a quote analysis of what the artist says about art "
    "to bob@example.com",

    "What does the artist say about art?",
]


for index, request in enumerate(test_cases, start=1):
    result = router.route(request)

    print(f"\nTEST {index}")
    print(f"Request: {request}")
    print(f"Result:  {result}")