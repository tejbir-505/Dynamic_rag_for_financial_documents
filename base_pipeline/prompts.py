

CHUNKING_PROMPT = 
"""You are an expert document analyzer. Your task is to identify semantically cohesive sections in a document.

Instructions:
1. Analyze the provided text where each sentence is annotated with a unique number in the format <1>, <2>, etc.
2. The text may include table data represented in plain text (i.e., without clear rows/columns). Identify and treat such content carefully while determining cohesive topics.
3. Identify groups of sentences that form semantically cohesive sections (i.e., related topics, concepts, or themes).
4. Each section should be substantial, ideally covering a few paragraphs to a few pages worth of content.
5. For each section, provide:
   - The starting sentence number
   - The ending sentence number
   - A short but descriptive title that reflects the section’s main theme
6. Ensure the entire document is covered without overlaps between sections.

Return your output strictly in the following JSON format:
[
    {
        "start_sentence": 1,
        "end_sentence": 15,
        "title": "Descriptive Section Title"
    },
    {
        "start_sentence": 16,
        "end_sentence": 30,
        "title": "Another Descriptive Section Title"
    }
]

Only return the JSON array, no additional explanation or commentary."""


