"""
Model runner for local LLM integration via Ollama
"""

import requests
import json
import logging
from typing import Dict, Optional
from config import OLLAMA_HOST, OLLAMA_MODEL

# Set up logging
logger = logging.getLogger(__name__)


class OllamaModelRunner:
    """Wrapper for calling local Ollama LLM"""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        self.generate_url = f"{self.host}/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Ollama not available: {e}")
            return False

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate response from local LLM

        Args:
            prompt: User question/prompt
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dict with response and metadata
        """

        # Check if service is available
        if not self.is_available():
            return {
                "success": False,
                "error": "Ollama service not available. Please ensure Ollama is running.",
                "response": None,
            }

        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Ollama API payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,  # Get complete response
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        try:
            logger.info(f"Generating response with model: {self.model}")

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60,  # 60 seconds timeout for generation
            )

            if response.status_code == 200:
                result = response.json()

                return {
                    "success": True,
                    "response": result.get("response", "").strip(),
                    "model": self.model,
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_time": result.get("total_duration", 0)
                    / 1e9,  # Convert to seconds
                    "error": None,
                }
            else:
                error_msg = (
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "response": None}

        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "response": None}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "response": None}

    def health_check(self) -> Dict:
        """Health check for the model service"""
        try:
            available = self.is_available()
            if available:
                # Test with a simple prompt
                test_result = self.generate_response("Hello!", max_tokens=10)
                return {
                    "status": "healthy" if test_result["success"] else "degraded",
                    "model": self.model,
                    "host": self.host,
                    "available": available,
                    "test_response": test_result["success"],
                }
            else:
                return {
                    "status": "unhealthy",
                    "model": self.model,
                    "host": self.host,
                    "available": False,
                    "error": "Service not available",
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance
model_runner = OllamaModelRunner()
