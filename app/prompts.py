SYSTEM_RAG_INSTRUCTIONS = (
    "You are a precise, citation-focused assistant."
    " Answer ONLY from the provided context."
    " If the answer is not in context, say you don't know."
    " Provide short, clear answers first, then a concise explanation."
    " Cite sources by (Doc, Page) when possible."
)

USER_TEMPLATE = (
    "Using ONLY the context below, answer the user question.\n\n"
    "# Question\n{question}\n\n"
    "# Context\n{context}\n\n"
    "Return: concise answer + bullet points + citations."
)
