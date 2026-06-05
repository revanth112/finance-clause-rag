from openai import AzureOpenAI
from app.core.config import settings
from typing import List, Dict

class LLMService:
    """
    Azure OpenAI GPT-4o-mini service for grounded answer generation.
    Answers are strictly grounded in retrieved chunks only.
    """
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate a grounded answer from retrieved chunks.
        Only uses information present in the context.
        """
        context = "\n\n".join([
            f"[Source {i+1}: {c.get('section_title', 'Unknown')}]\n{c.get('text', '')}"
            for i, c in enumerate(context_chunks)
        ])

        system_prompt = (
            "You are a Finance Clause Assistant. "
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say 'Not found in provided documents.' "
            "Always cite the source number (e.g. [Source 1]) in your answer."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        return response.choices[0].message.content

llm_service = LLMService()
