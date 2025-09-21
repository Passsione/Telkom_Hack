import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import google.generativeai as genai
from PIL import Image
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiIntegration:
    """
    Google GenAI SDK Integration for T-Help Technical Support Chatbot
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Google GenAI client"""
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        self.system_prompt = self._create_telkom_system_prompt()
        
        # Initialize models with the system prompt
        self.text_model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=self.system_prompt
        )
        self.vision_model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=self.system_prompt
        )
        
        logger.info("✅ Gemini integration initialized successfully")

    def _create_telkom_system_prompt(self) -> str:
        """Create specialized system prompt for Telkom technical support"""
        return """You are T-Help, an expert Telkom technical support assistant. Your role is to help customers troubleshoot technical issues with their Telkom services including:
- Internet connectivity problems (Wi-Fi, ADSL/Fiber)
- Mobile network problems
- Device and email configuration
- Network speed and performance issues
When analyzing files/images:
- Screenshots: Identify error messages and provide solutions.
- Bills/Documents: Help understand technical service details.
- Videos: Describe technical procedures shown.
Always be helpful, patient, and professional. Provide step-by-step instructions and ask clarifying questions. Respond in the same language as the customer's query."""

    def _format_history(self, history: List[Dict]) -> List[Dict]:
        """Formats the chat history for the Gemini API."""
        formatted = []
        for msg in history:
            role = 'user' if msg['type'] == 'user' else 'model'
            # Use text_content if available (for file uploads with text), otherwise use content
            content = msg.get('text_content', msg.get('content', ''))
            formatted.append({'role': role, 'parts': [content]})
        return formatted

    async def process_text_message(
        self, 
        message: str, 
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Process text message and generate AI response"""
        try:
            chat_session = self.text_model.start_chat(
                history=self._format_history(conversation_history or [])
            )
            response = await chat_session.send_message_async(message)
            
            return {
                'success': True,
                'response': response.text,
                'model_used': 'gemini-2.5-flash',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing text with Gemini: {str(e)}")
            return self._create_error_response(e)

    async def process_image_message(
        self, 
        image_path: str, 
        text_message: str = ""
    ) -> Dict[str, Any]:
        """Process image with optional text message"""
        try:
            img = Image.open(image_path)
            prompt = text_message or "Analyze this image for technical issues or information that might help with Telkom technical support."
            
            response = await self.vision_model.generate_content_async([prompt, img])
            
            return {
                'success': True,
                'response': response.text,
                'model_used': 'gemini-2.5-flash',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            return self._create_error_response(e)

    async def process_pdf_document(
        self, 
        pdf_path: str, 
        text_message: str = ""
    ) -> Dict[str, Any]:
        """Process PDF document by extracting text"""
        try:
            pdf_text = self._extract_pdf_text(pdf_path)
            combined_message = f"""
            User message: {text_message}
            
            PDF Content Summary:
            {pdf_text[:4000]}
            
            Please analyze this document and help with any Telkom technical issues mentioned.
            """
            return await self.process_text_message(message=combined_message)
            
        except Exception as e:
            logger.error(f"Error processing PDF with Gemini: {str(e)}")
            return self._create_error_response(e)

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ''
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return "Could not extract text from PDF."

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': str(error),
            'response': "I'm sorry, I'm having trouble processing your request right now. Please try again or contact Telkom support directly.",
            'timestamp': datetime.now().isoformat()
        }