"""AG News class IDs follow the imported Parquet ClassLabel metadata."""
SETTINGS = {
    "name": "AG-News",
    "text_column": "text",
    "labels": ["World", "Sports", "Business", "Sci/Tech"],
    "validation_fraction": 0.1,
    "split_seed": 42,
    "definitions": [
        "News primarily about international affairs, politics, diplomacy, conflict, or major public events, rather than sports, business, or scientific and technological developments.",
        "News primarily about athletic competitions, teams, players, results, or sporting events; sports is the central topic rather than an incidental reference.",
        "News primarily about companies, markets, finance, commerce, employment, or the economy; financial or commercial consequences dominate over scientific or technical details.",
        "News primarily about scientific discoveries, research, computing, engineering, or technology; scientific or technical developments dominate over stock prices or commercial earnings.",
    ],
    "generation_instructions": (
        "Write English news classification examples, NOT user requests or assistant replies. "
        "Each example should resemble a short headline followed by a brief news description, usually 25–65 words. "
        "Use varied locations, entities, events, and sentence structures. Vary headline wording and reporting style. "
        "Write plausible fictional news snippets, not claims of verified current reporting. "
        "One category must clearly dominate; avoid ambiguous multi-topic stories. "
        "Do not prepend the category name, a label annotation, or an explanation."
    ),
    "max_characters": 1600,
}
