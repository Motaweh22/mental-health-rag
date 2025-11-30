from llm_client_unsloth import generate_llm_answer

def build_rephrase_prompt(query: str, retrieved_answer: str):
    return f"""
You are a careful, empathetic mental health assistant.

INSTRUCTIONS:
- Rephrase the 'Original retrieved answer' in a calm, supportive, and non-judgmental tone.
- DO NOT invent facts or new information that are not present in the original answer.
- If the user's content contains signs of severe distress or self-harm, gently and clearly encourage them to seek immediate help from a licensed mental health professional or crisis hotline.
- Keep the content factual, brief if possible, and encourage seeking help when necessary.

User question:
{query}

Original retrieved answer:
{retrieved_answer}

Rewrite below:
"""

def rephrase_answer(query: str, retrieved_answer: str, max_new_tokens: int = 256, temperature: float = 0.25, top_p: float = 0.9, system_prompt: str = None, enable_safety_prompt: bool = True):
    # optional system_prompt and safety are handled by caller; keep interface flexible
    prompt = build_rephrase_prompt(query, retrieved_answer)
    return generate_llm_answer(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
