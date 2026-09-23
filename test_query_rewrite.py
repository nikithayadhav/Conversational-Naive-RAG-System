from backend.app.services.query_rewrite_service import rewrite_question


history = [
    {
        "role": "user",
        "content": "What is FinLeafe?"
    },
    {
        "role": "assistant",
        "content": "FinLeafe is a digital finance platform."
    }
]


question = "What are its security features?"


rewritten_question = rewrite_question(
    question=question,
    history=history
)


print("Original Question:")
print(question)

print("\nRewritten Question:")
print(rewritten_question)