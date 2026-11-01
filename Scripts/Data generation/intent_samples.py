import ollama

def main(params: dict):
    if "example_num" not in params.keys():
        raise KeyError("'example_num' must be passed in input params")
    if "intent_label" not in params.keys():
        raise KeyError("'intent_label' must be passed in input params")
    if "temperature" not in params.keys():
        raise KeyError("'temperature' must be passed in input params")
    if "seed" not in params.keys():
        raise KeyError("'seed' must be passed in input params")
    prompt = f"""Generate {params["example_num"]} synthetic training examples for an English intent classifier.

    Intent: {params["intent_label"].replace("_", " ")}

    Scope:
    - Every message must clearly express the specified intent.
    - Exclude messages whose primary purpose belongs to a different intent.
    - Avoid combining multiple intents in one message.
    - A related topic alone is not enough; the user's underlying goal must
    match the specified intent.

    Requirements:
    - Write realistic user messages, not assistant responses.
    - Vary sentence structure, vocabulary, and length.
    - Include both direct requests and indirect statements.
    - Avoid near-duplicates.
    - Do not include class annotations or explanations alongside the messages.

    Return only a JSON object in this format:
    {{
        "examples": [
        "first user message",
        "second user message"
        ]
    }}"""
    response = ollama.chat(
        model = "qwen2.5:3b",
        messages = [{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": params["temperature"], "seed": params["seed"]},
    )
    return response.message.content
if __name__ == "__main__":
    params = {
        "example_num": 10,
        "intent_label": "accept_reservations",
        "temperature": 0.8,
        "seed": 42,
    }
    print(main(params))