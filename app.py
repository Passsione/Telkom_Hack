from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import json
import base64
from werkzeug.utils import secure_filename
import uuid
import asyncio
from openai_integration import OpenAIIntegration

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wav', 'mp3', 'ogg', 'webm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI Integration
try:
    openai_client = OpenAIIntegration()
    print("✅ OpenAI integration initialized successfully")
    if not openai_client.api_available:
        print("⚠️  OpenAI API has quota/billing issues - running in fallback mode")
except Exception as e:
    print(f"❌ OpenAI integration failed: {e}")
    openai_client = None

# In-memory chat storage (replace with database later)
chat_sessions = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    """Determine file type based on extension"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in ['jpg', 'jpeg', 'png', 'gif']:
        return 'image'
    elif ext in ['mp4', 'avi', 'mov']:
        return 'video'
    elif ext in ['wav', 'mp3', 'ogg', 'webm']:
        return 'audio'
    elif ext == 'pdf':
        return 'pdf'
    else:
        return 'document'

async def generate_ai_response(message_content, message_type='text', file_path=None, session_id=None):
    """
    Generate AI response using OpenAI integration
    """
    if not openai_client:
        # Fallback to demo responses if OpenAI is not available
        return simulate_ai_response_fallback(message_content, message_type)
    
    try:
        # Get conversation history for context
        conversation_history = []
        if session_id and session_id in chat_sessions:
            # Convert chat history to OpenAI format
            for msg in chat_sessions[session_id][-6:]:  # Last 6 messages for context
                if msg['type'] == 'user':
                    conversation_history.append({
                        "role": "user", 
                        "content": msg.get('text_content', msg['content'])
                    })
                elif msg['type'] == 'ai':
                    conversation_history.append({
                        "role": "assistant", 
                        "content": msg['content']
                    })
        
        # Process based on message type
        if message_type == 'image' and file_path:
            response = await openai_client.process_image_message(
                image_path=file_path,
                text_message=message_content
            )
        elif message_type == 'audio' and file_path:
            response = await openai_client.process_audio_message(file_path)
        elif message_type == 'pdf' and file_path:
            response = await openai_client.process_pdf_document(
                pdf_path=file_path,
                text_message=message_content
            )
        else:
            # Text message or file with text
            response = await openai_client.process_text_message(
                message=message_content,
                conversation_history=conversation_history
            )
        
        if response['success']:
            return response['response']
        else:
            return f"I apologize, I'm experiencing technical difficulties: {response.get('error', 'Unknown error')}. Please try again or contact Telkom support directly."
    
    except Exception as e:
        print(f"AI Response Error: {e}")
        return "I'm currently experiencing technical difficulties. Please try again in a moment or contact Telkom support directly for immediate assistance."

def simulate_ai_response_fallback(message, message_type='text', user_language='en'):
    """
    Fallback AI response simulation when OpenAI is not available
    """
    responses = {
        'en': {
            'greeting': "Hello! I'm T-Help, your Telkom technical assistant. I'm currently running in demo mode. For full AI capabilities, please ensure your OpenAI API key is configured. How can I help you today?",
            'internet_issue': "I understand you're having internet connectivity issues. Let me help you troubleshoot this step by step:\n\n1. Please check if all cables are securely connected\n2. Try restarting your router by unplugging it for 30 seconds\n3. Check if other devices can connect to the internet\n\nCan you tell me which step you'd like to try first?",
            'file_received': f"I've received your {message_type}. In full mode, I would analyze this content for you. Currently running in demo mode.",
            'voice_note': "I've received your voice message. In full mode, I would transcribe and analyze your audio. Currently running in demo mode.",
            'default': "I'm here to help with your Telkom technical issues. Currently running in demo mode - for full AI capabilities, please configure your OpenAI API key. Can you describe the problem you're experiencing?"
        }
    }
    
    message_lower = message.lower() if isinstance(message, str) else ""
    
    if message_type == 'voice':
        return responses['en']['voice_note']
    elif message_type in ['image', 'video', 'pdf', 'document']:
        return responses['en']['file_received']
    elif any(word in message_lower for word in ['internet', 'connection', 'wifi', 'slow', 'not working']):
        return responses['en']['internet_issue']
    elif any(word in message_lower for word in ['hello', 'hi', 'help', 'start']):
        return responses['en']['greeting']
    else:
        return responses['en']['default']

@app.route('/')
def index():
    """Home page with chat interface"""
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle sending messages (text, files, voice notes)"""
    try:
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Get text message (can be combined with file)
        text_message = request.form.get('message', '').strip()
        
        # Handle voice note (base64 audio data)
        if 'voice_data' in request.files:
            voice_file = request.files['voice_data']
            if voice_file:
                # Save voice note with proper audio format
                voice_filename = f"voice_{uuid.uuid4()}.webm"
                voice_path = os.path.join(UPLOAD_FOLDER, voice_filename)
                voice_file.save(voice_path)
                
                user_message = {
                    'id': len(chat_sessions[session_id]) + 1,
                    'type': 'user',
                    'content': "🎵 Voice message",
                    'message_type': 'voice',
                    'file_path': voice_path,
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                chat_sessions[session_id].append(user_message)
                
                # Generate AI response using OpenAI
                ai_response_text = asyncio.run(generate_ai_response(
                    message_content="Voice message received",
                    message_type='audio',
                    file_path=voice_path,
                    session_id=session_id
                ))
                
                ai_message = {
                    'id': len(chat_sessions[session_id]) + 1,
                    'type': 'ai',
                    'content': ai_response_text,
                    'message_type': 'text',
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                chat_sessions[session_id].append(ai_message)
                
                return jsonify({
                    'status': 'success',
                    'user_message': user_message,
                    'ai_message': ai_message,
                    'session_id': session_id
                })
        
        # Handle file upload (can be combined with text)
        elif 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
                file.save(file_path)
                
                file_type = get_file_type(filename)
                
                # Create message content (file + optional text)
                if text_message:
                    content = f"{text_message}\n📎 {filename}"
                    message_for_ai = text_message
                else:
                    content = f"📎 {filename}"
                    message_for_ai = f"Analyze this {file_type}: {filename}"
                
                # Add user message
                user_message = {
                    'id': len(chat_sessions[session_id]) + 1,
                    'type': 'user',
                    'content': content,
                    'message_type': file_type,
                    'file_path': file_path,
                    'text_content': text_message if text_message else None,
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                chat_sessions[session_id].append(user_message)
                
                # Generate AI response using OpenAI
                ai_response_text = asyncio.run(generate_ai_response(
                    message_content=message_for_ai,
                    message_type=file_type,
                    file_path=file_path,
                    session_id=session_id
                ))
                
                ai_message = {
                    'id': len(chat_sessions[session_id]) + 1,
                    'type': 'ai',
                    'content': ai_response_text,
                    'message_type': 'text',
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                chat_sessions[session_id].append(ai_message)
                
                return jsonify({
                    'status': 'success',
                    'user_message': user_message,
                    'ai_message': ai_message,
                    'session_id': session_id
                })
        
        # Handle text-only message
        elif text_message:
            # Add user message
            user_message = {
                'id': len(chat_sessions[session_id]) + 1,
                'type': 'user',
                'content': text_message,
                'message_type': 'text',
                'timestamp': datetime.now().strftime('%H:%M')
            }
            chat_sessions[session_id].append(user_message)
            
            # Generate AI response using OpenAI
            ai_response_text = asyncio.run(generate_ai_response(
                message_content=text_message,
                message_type='text',
                session_id=session_id
            ))
            
            ai_message = {
                'id': len(chat_sessions[session_id]) + 1,
                'type': 'ai',
                'content': ai_response_text,
                'message_type': 'text',
                'timestamp': datetime.now().strftime('%H:%M')
            }
            chat_sessions[session_id].append(ai_message)
            
            return jsonify({
                'status': 'success',
                'user_message': user_message,
                'ai_message': ai_message,
                'session_id': session_id
            })
        
        else:
            return jsonify({'status': 'error', 'message': 'No content provided'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_chat_history/<session_id>')
def get_chat_history(session_id):
    """Get chat history for a session"""
    if session_id in chat_sessions:
        return jsonify({
            'status': 'success',
            'messages': chat_sessions[session_id]
        })
    else:
        return jsonify({
            'status': 'success',
            'messages': []
        })

@app.route('/api_status')
def api_status():
    """Check OpenAI API status and available models"""
    if openai_client:
        status = asyncio.run(openai_client.get_model_status() if hasattr(openai_client, 'get_model_status') else {'success': True, 'status': 'Connected'})
        return jsonify(status)
    else:
        return jsonify({
            'success': False,
            'error': 'OpenAI client not initialized',
            'message': 'Please set OPENAI_API_KEY environment variable'
        })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)