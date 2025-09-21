from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import os
import json
import base64
from werkzeug.utils import secure_filename
import uuid
import asyncio
import nest_asyncio
from gemini_integration import GeminiIntegration 

# Allow nested event loops
nest_asyncio.apply()

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wav', 'mp3', 'ogg', 'webm'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Gemini Integration
try:
    gemini_client = GeminiIntegration()
except Exception as e:
    print(f"❌ Gemini integration failed: {e}")
    gemini_client = None

# In-memory chat storage
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

def generate_ai_response(message_content, message_type='text', file_path=None, session_id=None):
    """
    Generate AI response using Gemini integration - SYNC VERSION
    """
    if not gemini_client:
        return "The AI assistant is currently unavailable due to a configuration error."
    
    try:
        conversation_history = chat_sessions.get(session_id, [])[-6:]
        
        # Get or create event loop
        # try:
        loop = asyncio.get_event_loop()
        # except RuntimeError:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        
        if message_type == 'image' and file_path:
            response_data = loop.run_until_complete(
                gemini_client.process_image_message(
                    image_path=file_path,
                    text_message=message_content,
                    conversation_history=conversation_history
                )
            )
        elif message_type in ['audio', 'video'] and file_path:
            response_data = loop.run_until_complete(
                gemini_client.process_audio_message(
                    audio_path=file_path,
                    text_message=message_content,
                    conversation_history=conversation_history
                )
            )
        elif message_type == 'pdf' and file_path:
            response_data = loop.run_until_complete(
                gemini_client.process_pdf_document(
                    pdf_path=file_path,
                    text_message=message_content,
                    conversation_history=conversation_history
                )
            )
        else:  # Text message
            response_data = loop.run_until_complete(
                gemini_client.process_text_message(
                    message=message_content,
                    conversation_history=conversation_history
                )
            )
        
        return response_data['response'] if response_data['success'] else response_data['error']
    
    except Exception as e:
        print(f"AI Response Error: {e}")
        return "I'm experiencing technical difficulties. Please try again or contact Telkom support."

@app.route('/')
def index():
    """Home page with chat interface"""
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle sending messages (text, files, voice notes)"""
    try:
        session_id = request.form.get('session_id') or str(uuid.uuid4())
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        text_message = request.form.get('message', '').strip()
        file = request.files.get('file') or request.files.get('voice_data')

        user_message = None
        ai_response_text = None

        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
                file.save(file_path)
                
                file_type = get_file_type(filename)
                
                # Create user message for display
                if file_type == 'audio':
                     display_content = "🎵 Voice message"
                else:
                     display_content = f"📎 {filename}"
                
                if text_message:
                    display_content = f"{text_message} \n {display_content}"

                user_message = {
                    'id': len(chat_sessions[session_id]) + 1,
                    'type': 'user',
                    'content': display_content,
                    'message_type': file_type,
                    'file_path': file_path,
                    'text_content': text_message,
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                chat_sessions[session_id].append(user_message)
                
                # Generate AI response
                ai_response_text = generate_ai_response(
                    message_content=text_message,
                    message_type=file_type,
                    file_path=file_path,
                    session_id=session_id
                )
            else:
                return jsonify({'status': 'error', 'message': 'File type not allowed.'})

        elif text_message:
            user_message = {
                'id': len(chat_sessions[session_id]) + 1,
                'type': 'user',
                'content': text_message,
                'message_type': 'text',
                'timestamp': datetime.now().strftime('%H:%M')
            }
            chat_sessions[session_id].append(user_message)
            
            ai_response_text = generate_ai_response(
                message_content=text_message,
                message_type='text',
                session_id=session_id
            )
        
        else:
            return jsonify({'status': 'error', 'message': 'No content provided'})

        if ai_response_text:
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

        return jsonify({'status': 'error', 'message': 'Could not generate AI response.'})
            
    except Exception as e:
        print(f"ERROR in send_message: {e}")
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

# Run the app
if __name__ == '__main__':
    app.run(debug=True)