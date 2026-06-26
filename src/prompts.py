ANSWER_FROM_CONTEXT = """Answer using ONLY the context provided. Do not use outside knowledge.
If the context does not contain enough to answer, reply with exactly: NOT_FOUND

Context:
{context}

Question: {question}
Answer:"""

WEB_SYNTHESIS = """Answer using only the numbered search results. Cite sources inline as [1], [2], etc.
If the results don't answer the question, say so plainly.

Results:
{results}

Question: {question}
Answer:"""
