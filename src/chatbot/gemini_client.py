import os
import google.generativeai as genai
from typing import Optional

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with the API key.
        
        Args:
            api_key: Google Gemini API key. If None, will try to read from GEMINI_API_KEY environment variable.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            print("Warning: No Gemini API key provided. Please set the GEMINI_API_KEY environment variable.")
        else:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Set up the model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Set default parameters
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured with an API key."""
        return self.api_key is not None
    
    def ask_health_question(self, question: str) -> str:
        """Ask a health-related question to Gemini.
        
        Args:
            question: The health-related question to ask.
            
        Returns:
            str: The response from Gemini or an error message.
        """
        if not self.is_configured():
            return "Error: Gemini API is not configured. Please provide an API key."
        
        try:
            # Add health context to the prompt
            prompt = f"""You are a helpful health information assistant. 
            Provide accurate, evidence-based information about health topics.
            Note that you are NOT providing medical diagnosis or treatment advice, 
            only general health information.
            
            User's question: {question}
            
            Please provide a helpful, informative, and accurate response:"""
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
        except Exception as e:
            return f"Error querying Gemini API: {str(e)}"

# Example usage
if __name__ == "__main__":
    client = GeminiClient()
    if client.is_configured():
        question = "Làm sao để phòng tránh COVID?"
        response = client.ask_health_question(question)
        print(f"Question: {question}")
        print(f"Response: {response}")
    else:
        print("Please provide a Gemini API key to test.") 