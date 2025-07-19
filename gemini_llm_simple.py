# gemini_llm_simple.py
import google.generativeai as genai
from typing import Optional, List
import logging

class GeminiLLM:
    """Classe simple pour intégrer Gemini sans hériter de LangChain LLM"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = 0.7
        self.max_tokens = 1000
        
        # Configuration de Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def invoke(self, prompt: str) -> str:
        """Génère du contenu avec Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Erreur Gemini API: {e}")
            return f"Erreur lors de la génération: {str(e)}"
    
    def generate(self, prompt: str) -> str:
        """Alias pour invoke"""
        return self.invoke(prompt)