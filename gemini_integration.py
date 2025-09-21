import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import google.generativeai as genai
from PIL import Image
import PyPDF2
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiIntegration:
    """
    Google GenAI SDK Integration for T-Help Technical Support Chatbot
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Google GenAI client"""
        self.supported_audio_formats = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.ogg': 'audio/ogg',
            '.webm': 'audio/webm'
        }
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        self.system_prompt = self._create_telkom_system_prompt()
        
        # Initialize single model for all operations
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
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
            content = msg.get('text_content', msg.get('content', ''))
            formatted.append({'role': role, 'parts': [content]})
        return formatted

    
    def _validate_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio file format and size"""
        try:
            if not os.path.exists(audio_path):
                return {'valid': False, 'error': f"Audio file not found: {audio_path}"}
            
            # Check file size (Gemini has file size limits)
            file_size = os.path.getsize(audio_path)
            max_size = 20 * 1024 * 1024  # 20MB limit for audio files
            
            if file_size > max_size:
                return {'valid': False, 'error': f"Audio file too large: {file_size / 1024 / 1024:.1f}MB (max 20MB)"}
            
            if file_size == 0:
                return {'valid': False, 'error': "Audio file is empty"}
            
            # Check file extension
            _, ext = os.path.splitext(audio_path.lower())
            if ext not in self.supported_audio_formats:
                return {'valid': False, 'error': f"Unsupported audio format: {ext}. Supported: {', '.join(self.supported_audio_formats.keys())}"}
            
            # Get MIME type
            mime_type = self.supported_audio_formats.get(ext)
            
            return {
                'valid': True, 
                'mime_type': mime_type,
                'size': file_size,
                'extension': ext
            }
            
        except Exception as e:
            return {'valid': False, 'error': f"Error validating audio file: {str(e)}"}

    async def process_audio_message(
        self,
        audio_path: str,
        text_message: str = "",
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Process audio file by uploading it to the File API with improved error handling."""
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Validate audio file first
            validation = self._validate_audio_file(audio_path)
            if not validation['valid']:
                logger.error(f"Audio validation failed: {validation['error']}")
                return self._create_error_response(Exception(validation['error']))
            
            logger.info(f"Audio file validated: {validation['size']} bytes, type: {validation['mime_type']}")
            
            # Upload file with explicit MIME type
            try:
                audio_file = genai.upload_file(
                    path=audio_path,
                    mime_type=validation['mime_type']
                )
                logger.info(f"Audio file uploaded successfully: {audio_file.name}")
            except Exception as upload_error:
                logger.error(f"Upload failed: {str(upload_error)}")
                return self._create_error_response(Exception(f"Failed to upload audio file: {str(upload_error)}"))
            
            # Wait for processing with better status monitoring
            max_wait = 60  # Increased wait time for larger files
            waited = 0
            check_interval = 3
            
            logger.info("Waiting for audio file processing...")
            while audio_file.state.name == "PROCESSING" and waited < max_wait:
                time.sleep(check_interval)
                waited += check_interval
                try:
                    audio_file = genai.get_file(audio_file.name)
                    logger.info(f"Audio processing status: {audio_file.state.name} (waited {waited}s)")
                except Exception as status_error:
                    logger.error(f"Error checking file status: {str(status_error)}")
                    break
            
            # Check final state
            if audio_file.state.name == "PROCESSING":
                error_msg = f"Audio processing timeout after {max_wait}s"
                logger.error(error_msg)
                return self._create_error_response(Exception(error_msg))
            
            if audio_file.state.name == "FAILED":
                error_msg = "Audio file processing failed - file may be corrupted or in unsupported format"
                logger.error(error_msg)
                return self._create_error_response(Exception(error_msg))
            
            if audio_file.state.name != "ACTIVE":
                error_msg = f"Audio file not ready: {audio_file.state.name}"
                logger.error(error_msg)
                return self._create_error_response(Exception(error_msg))
            
            logger.info(f"✅ Audio file {audio_file.name} is now ACTIVE and ready for processing.")

            # Create chat session and send message
            chat_session = self.model.start_chat(
                history=self._format_history(conversation_history or [])
            )
            
            user_prompt = text_message or "Please transcribe and analyze this audio message. If you hear any Telkom technical support issues or questions, provide helpful troubleshooting assistance."
            
            response = chat_session.send_message([user_prompt, audio_file])
            
            # Clean up uploaded file (optional - Gemini will auto-delete after 48 hours)
            try:
                genai.delete_file(audio_file.name)
                logger.info(f"Cleaned up uploaded file: {audio_file.name}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up file: {cleanup_error}")
            
            return {
                'success': True,
                'response': response.text,
                'model_used': 'gemini-2.0-flash',
                'audio_info': {
                    'size': validation['size'],
                    'type': validation['mime_type'],
                    'processing_time': f"{waited}s"
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {str(e)}")
            return self._create_error_response(e)

    async def process_text_message(
        self, 
        message: str, 
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Process text message and generate AI response"""
        try:
            chat_session = self.model.start_chat(
                history=self._format_history(conversation_history or [])
            )
            response = chat_session.send_message(message)
            
            return {
                'success': True,
                'response': response.text,
                'model_used': 'gemini-2.0-flash',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing text with Gemini: {str(e)}")
            return self._create_error_response(e)

    async def process_image_message(
        self, 
        image_path: str, 
        text_message: str = "",
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Process image with optional text message and history"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = Image.open(image_path)
            
            chat_session = self.model.start_chat(
                history=self._format_history(conversation_history or [])
            )

            user_prompt = text_message or "Please analyze this image for any Telkom technical support issues or error messages."
            message_parts = [user_prompt, img]
            
            response = chat_session.send_message(message_parts)
            
            return {
                'success': True,
                'response': response.text,
                'model_used': 'gemini-2.0-flash',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            return self._create_error_response(e)

    async def process_pdf_document(
        self, 
        pdf_path: str, 
        text_message: str = "",
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Process PDF document by extracting text"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            pdf_text = self._extract_pdf_text(pdf_path)
            combined_message = f"""
            User message: {text_message}
            
            PDF Content:
            {pdf_text[:8000]}
            
            Please analyze the document content and help with any Telkom technical issues mentioned.
            """
            
            return await self.process_text_message(
                message=combined_message, 
                conversation_history=conversation_history
            )
            
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