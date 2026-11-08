"""SST-2 class IDs follow the imported Parquet ClassLabel metadata."""
SETTINGS = {
    "name": "SST-2",
    "text_column": "sentence",
    "labels": ["negative", "positive"],
    "validation_fraction": None,
    "split_seed": 42,
    "definitions": [
        "A movie-review sentence or phrase expressing an overall unfavorable opinion of the film, its storytelling, performances, direction, or other cinematic qualities.",
        "A movie-review sentence or phrase expressing an overall favorable opinion of the film, its storytelling, performances, direction, or other cinematic qualities.",
    ],
    "generation_instructions": (
        "Write English movie-review sentiment examples, NOT customer-service requests or assistant replies. "
        "Include both short review phrases/fragments (roughly 2–8 words) and complete review sentences (roughly 9–30 words). "
        "The source task includes fragments, so do not turn every example into a polished full review. "
        "Vary aspects such as acting, pacing, dialogue, direction, cinematography, and emotional impact. "
        "Vary sentence structure, vocabulary, and explicitness; include some natural negation and qualified judgments. "
        "The overall sentiment must clearly match the assigned category; exclude neutral text and unresolved mixed sentiment. "
        "Avoid generic product reviews, repetitive praise/complaint templates, and deliberate label-word stuffing. "
        "Do not prepend a sentiment label, star rating, or explanation."
    ),
    "max_characters": 600,
}
