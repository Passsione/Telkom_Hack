import openai
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import base64
from io import BytesIO
from PIL import Image
import speech_recognition as sr
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIIntegration:
    """
    OpenAI Integration for T-Help Technical Support Chatbot
    Supports latest GPT models including GPT-5, GPT-4.1 series
    """
    
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configurations - using latest available models
        self.models = {
            'primary': 'gpt-5',  # Latest and most capable
            'fallback': 'gpt-4.1',  # High performance fallback
            'cost_efficient': 'gpt-4.1-mini',  # Cost-efficient option
            'fast': 'gpt-4.1-nano',  # Fastest response
            'vision': 'gpt-5',  # For image analysis
            'audio': 'whisper-1'  # For speech recognition
        }
        
        # Telkom-specific system prompt
        self.system_prompt = self._create_telkom_system_prompt()
        
        # Language detection patterns
        self.language_patterns = {
            'afrikaans': ['ek', 'jy', 'dis', 'nie', 'wat', 'hoe', 'waar', 'wanneer', 'hoekom', 'asseblief', 'dankie'],
            'zulu': ['ngi', 'uku', 'ngi-', 'isi', 'aba', 'ama', 'sawubona', 'ngiyabonga', 'unjani', 'kunjani'],
            'english': ['the', 'and', 'you', 'how', 'what', 'where', 'when', 'why', 'please', 'thank']
        }

    def _create_telkom_system_prompt(self) -> str:
        """Create specialized system prompt for Telkom technical support"""
        return """You are T-Help, an expert Telkom technical support assistant. Your role is to help customers troubleshoot technical issues with their Telkom services including:

- Internet connectivity problems
- Wi-Fi and router issues  
- Mobile network problems
- ADSL/Fiber connection issues
- Email setup and configuration
- Device configuration
- Network speed and performance issues
- Billing and account-related technical queries

IMPORTANT GUIDELINES:
1. Always be helpful, patient, and professional
2. Provide step-by-step troubleshooting instructions
3. Ask clarifying questions when needed
4. Escalate complex issues to human agents when appropriate
5. Support multiple languages (English, Afrikaans, Zulu)
6. Keep responses concise but comprehensive
7. Always prioritize customer safety and data security

When analyzing files/images:
- Screenshots: Identify error messages and provide solutions
- Network configs: Analyze settings and suggest corrections  
- Bills/Documents: Help understand technical service details
- Videos: Describe technical procedures shown

Respond in the same language as the customer's query."""

    def detect_language(self, text: str) -> str:
        """Detect the primary language of the input text"""
        if not text:
            return 'english'
        
        text_lower = text.lower()
        language_scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            language_scores[lang] = score
        
        detected_lang = max(language_scores, key=language_scores.get)
        return detected_lang if language_scores[detected_lang] > 0 else 'english'

    async def process_text_message(
        self, 
        message: str, 
        conversation_history: List[Dict] = None,
        model: str = None,
        language: str = None
    ) -> Dict[str, Any]:
        """
        Process text message and generate AI response
        """
        try:
            # Detect language if not provided
            if not language:
                language = self.detect_language(message)
            
            # Choose appropriate model
            model_name = model or self.models['primary']
            
            # Prepare conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Keep last 10 messages for context
            
            messages.append({"role": "user", "content": message})
            
            # Make API call
            response = await self._make_api_call(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                'success': True,
                'response': response['choices'][0]['message']['content'],
                'model_used': model_name,
                'detected_language': language,
                'usage': response.get('usage', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing text message: {str(e)}")
            return self._create_error_response(e)

    async def process_image_message(
        self, 
        image_path: str, 
        text_message: str = "",
        model: str = None
    ) -> Dict[str, Any]:
        """
        Process image with optional text message
        """
        try:
            model_name = model or self.models['vision']
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            # Prepare message content
            content = []
            if text_message:
                content.append({"type": "text", "text": text_message})
            else:
                content.append({"type": "text", "text": "Analyze this image for technical issues or information that might help with Telkom technical support."})
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
            
            response = await self._make_api_call(
                model=model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                'success': True,
                'response': response['choices'][0]['message']['content'],
                'model_used': model_name,
                'message_type': 'image_analysis',
                'usage': response.get('usage', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return self._create_error_response(e)

    async def process_audio_message(self, audio_path: str) -> Dict[str, Any]:
        """
        Process voice message using Whisper for speech-to-text
        """
        try:
            # Transcribe audio
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.models['audio'],
                    file=audio_file,
                    response_format="text"
                )
            
            # Process the transcribed text
            text_response = await self.process_text_message(transcript)
            
            return {
                'success': True,
                'transcription': transcript,
                'response': text_response['response'],
                'model_used': f"{self.models['audio']} + {text_response['model_used']}",
                'detected_language': text_response.get('detected_language'),
                'message_type': 'voice_message',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return self._create_error_response(e)

    async def process_pdf_document(
        self, 
        pdf_path: str, 
        text_message: str = ""
    ) -> Dict[str, Any]:
        """
        Process PDF document and extract relevant information
        """
        try:
            # Extract text from PDF
            pdf_text = self._extract_pdf_text(pdf_path)
            
            # Combine with user message
            combined_message = f"""
            User message: {text_message}
            
            PDF Content Summary:
            {pdf_text[:2000]}  # Limit to first 2000 characters
            
            Please analyze this document and help with any Telkom technical issues mentioned.
            """
            
            response = await self.process_text_message(
                message=combined_message,
                model=self.models['primary']
            )
            
            return {
                'success': True,
                'response': response['response'],
                'model_used': response['model_used'],
                'message_type': 'pdf_analysis',
                'pdf_text_length': len(pdf_text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return self._create_error_response(e)

    async def _make_api_call(self, **kwargs) -> Dict[str, Any]:
        """
        Make API call with fallback models
        """
        models_to_try = [
            kwargs.get('model', self.models['primary']),
            self.models['fallback'],
            self.models['cost_efficient']
        ]
        
        for model in models_to_try:
            try:
                kwargs['model'] = model
                response = self.client.chat.completions.create(**kwargs)
                return response.model_dump()
            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)}")
                if model == models_to_try[-1]:
                    raise e
                continue

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return "Could not extract text from PDF."

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': str(error),
            'response': "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment, or contact Telkom support directly for immediate assistance.",
            'timestamp': datetime.now().isoformat()
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Check the status of available models"""
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models.data]
            
            status = {}
            for category, model_name in self.models.items():
                status[category] = {
                    'model': model_name,
                    'available': model_name in available_models
                }
            
            return {
                'success': True,
                'model_status': status,
                'total_available': len(available_models)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }