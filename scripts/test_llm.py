from app.agent.llm import LocalLLM


llm = LocalLLM()


response = llm.generate(
    "In one short sentence, explain what a movie subtitle is."
)


print("\nOllama response:\n")
print(response)


assert response.strip()

print("\n" + "=" * 70)
print("LOCAL LLM TEST PASSED")
print("=" * 70)