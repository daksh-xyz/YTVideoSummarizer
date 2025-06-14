import os
import re
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import streamlit as st
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, set_key
from streamlit_option_menu import option_menu
from bs4 import BeautifulSoup
import update_cookies

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
CHATS_DIR = Path("./chats")
ENV_FILE = ".env"
DEFAULT_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 7000
CHUNK_OVERLAP = 1000
MAX_TOKENS = 8000

class YouTubeSummarizer:
    def __init__(self):
        self.groq_client = None
        self._initialize_client()
        self._ensure_chats_folder()

    def _initialize_client(self):
        """Initialize the Groq client with API key"""
        try:
            api_key = self._load_environment()
            self.groq_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        except Exception as e:
            st.error(f"Error initializing API client: {str(e)}")
            st.stop()

    def _load_environment(self) -> str:
        """Load and return the API key from environment variables"""
        env_path = Path(__file__).parent / ENV_FILE
        if env_path.exists():
            load_dotenv(env_path)
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return api_key

    def _ensure_chats_folder(self):
        """Create chats folder if it doesn't exist"""
        CHATS_DIR.mkdir(exist_ok=True)

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]
        
        youtube_url = youtube_url.strip()
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        raise ValueError("Could not extract video ID from URL")

    def get_transcript(self, youtube_url: str, preferred_language: str = 'en') -> Tuple[Optional[str], Optional[str]]:
        """Get transcript from YouTube video in preferred language"""
        try:
            video_id = self.extract_video_id(youtube_url)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Get available transcripts info for user feedback
            available_languages = []
            for transcript in transcript_list:
                lang_info = f"{transcript.language} ({transcript.language_code})"
                if transcript.is_generated:
                    lang_info += " [Auto-generated]"
                available_languages.append(lang_info)
            
            st.info(f"Available transcripts: {', '.join(available_languages)}")
            
            selected_transcript = None
            
            # Strategy 1: Try to find transcript in preferred language (manually created first)
            try:
                selected_transcript = transcript_list.find_transcript([preferred_language])
                st.success(f"Found transcript in preferred language: {preferred_language}")
            except:
                pass
            
            # Strategy 2: If preferred language not found, try English as fallback
            if not selected_transcript and preferred_language != 'en':
                try:
                    selected_transcript = transcript_list.find_transcript(['en'])
                    st.warning(f"Preferred language ({preferred_language}) not available. Using English transcript.")
                except:
                    pass
            
            # Strategy 3: Try to get any manually created transcript
            if not selected_transcript:
                try:
                    selected_transcript = transcript_list.find_manually_created_transcript()
                    st.warning(f"Using manually created transcript in: {selected_transcript.language_code}")
                except:
                    pass
            
            # Strategy 4: Get any available transcript (including auto-generated)
            if not selected_transcript:
                try:
                    # Get the first available transcript, but prefer non-Arabic if possible
                    transcripts = list(transcript_list)
                    
                    # First, try to find non-Arabic transcripts
                    non_arabic_transcripts = [t for t in transcripts if t.language_code not in ['ar', 'ar-SA', 'ar-EG']]
                    if non_arabic_transcripts:
                        selected_transcript = non_arabic_transcripts[0]
                        st.warning(f"Using available transcript in: {selected_transcript.language_code}")
                    else:
                        # If only Arabic available, use it
                        selected_transcript = transcripts[0]
                        st.warning(f"Only Arabic transcript available, using: {selected_transcript.language_code}")
                        
                except Exception:
                    st.error("Could not find any transcripts for this video.")
                    return None, None
            
            # Fetch transcript content
            try:
                transcript_parts = selected_transcript.fetch()
                if not transcript_parts:
                    st.error("No transcript content found.")
                    return None, None
                    
                # Extract text from transcript parts
                full_transcript = " ".join([
                    part.get('text', '') if isinstance(part, dict) else getattr(part, 'text', '')
                    for part in transcript_parts
                ])
                
                if not full_transcript.strip():
                    st.error("Transcript content is empty.")
                    return None, None
                    
                return full_transcript, selected_transcript.language_code
                
            except Exception as e:
                st.error(f"Error fetching transcript content: {str(e)}")
                return None, None
                
        except (NoTranscriptFound, TranscriptsDisabled):
            st.error("Could not fetch transcript. The video might be private, age-restricted, or not have captions available.")
            return None, None
        except Exception as e:
            st.error(f"Invalid YouTube URL or error accessing video: {str(e)}")
            return None, None

    def get_available_transcripts(self, youtube_url: str) -> List[Dict[str, str]]:
        """Get list of available transcripts for a video"""
        try:
            video_id = self.extract_video_id(youtube_url)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            transcripts_info = []
            for transcript in transcript_list:
                transcripts_info.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return transcripts_info
        except Exception:
            return []

    @staticmethod
    def get_available_languages() -> Dict[str, str]:
        """Return dictionary of available languages"""
        return {
            'English': 'en',
            'Deutsch': 'de',
            'Italiano': 'it',
            'Español': 'es',
            'Français': 'fr',
            'Nederlands': 'nl',
            'Polski': 'pl',
            '日本語': 'ja',
            '中文': 'zh',
            'Русский': 'ru'
        }

    def _get_language_prompts(self, target_language: str) -> Dict[str, str]:
        """Get language-specific prompts"""
        language_prompts = {
            'en': {
                'title': 'TITLE',
                'overview': 'OVERVIEW',
                'key_points': 'KEY POINTS',
                'takeaways': 'MAIN TAKEAWAYS',
                'context': 'CONTEXT & IMPLICATIONS'
            },
            'de': {
                'title': 'TITEL',
                'overview': 'ÜBERBLICK',
                'key_points': 'KERNPUNKTE',
                'takeaways': 'HAUPTERKENNTNISSE',
                'context': 'KONTEXT & AUSWIRKUNGEN'
            },
            'it': { 
                'title': 'TITOLO',
                'overview': 'PANORAMICA',
                'key_points': 'PUNTI CHIAVE',
                'takeaways': 'CONCLUSIONI PRINCIPALI',
                'context': 'CONTESTO E IMPLICAZIONI'
            }
        }
        return language_prompts.get(target_language, language_prompts['en'])

    def _create_summary_prompts(self, text: str, target_language: str, mode: str = 'video') -> Tuple[str, str]:
        """Create optimized prompts for summarization"""
        prompts = self._get_language_prompts(target_language)
        
        base_system = f"""You are an expert content analyst and summarizer. Create a comprehensive 
        {mode}-style summary in {target_language}. Ensure all content is fully translated and culturally adapted 
        to the target language."""

        if mode == 'podcast':
            user_prompt = f"""Please provide a detailed podcast-style summary of the following content in {target_language}. 
            Structure your response as follows:

            🎙️ {prompts['title']}: Create an engaging title

            🎧 {prompts['overview']} (3-5 sentences):
            - Provide a detailed context and main purpose

            🔍 {prompts['key_points']}:
            - Deep dive into the main arguments
            - Include specific examples and anecdotes
            - Highlight unique perspectives and expert opinions

            📈 {prompts['takeaways']}:
            - List 5-7 practical insights
            - Explain their significance and potential impact

            🌐 {prompts['context']}:
            - Broader context discussion
            - Future implications and expert predictions

            Text to summarize: {text}

            Ensure the summary is comprehensive enough for someone who hasn't seen the original content."""
        else:
            user_prompt = f"""Please provide a detailed summary of the following content in {target_language}. 
            Structure your response as follows:

            🎯 {prompts['title']}: Create a descriptive title

            📝 {prompts['overview']} (2-3 sentences):
            - Provide a brief context and main purpose

            🔑 {prompts['key_points']}:
            - Extract and explain the main arguments
            - Include specific examples
            - Highlight unique perspectives

            💡 {prompts['takeaways']}:
            - List 3-5 practical insights
            - Explain their significance

            🔄 {prompts['context']}:
            - Broader context discussion
            - Future implications

            Text to summarize: {text}

            Ensure the summary is comprehensive enough for someone who hasn't seen the original content."""

        return base_system, user_prompt

    def summarize_content(self, transcript: str, language_code: str, 
                         model_name: str = DEFAULT_MODEL, mode: str = 'video') -> Optional[str]:
        """Create summary using LangChain text splitting and OpenAI API"""
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        text_chunks = text_splitter.split_text(transcript)
        
        # Generate intermediate summaries
        intermediate_summaries = []
        
        for i, chunk in enumerate(text_chunks):
            system_prompt = f"""You are an expert content summarizer. Create a detailed 
            summary of section {i+1} in {language_code}. Maintain important details, arguments, 
            and connections. This summary will later be part of a comprehensive final summary."""

            user_prompt = f"""Create a detailed summary of the following section. 
            Maintain all important information, arguments, and connections.
            Pay special attention to:
            - Main topics and arguments
            - Important details and examples
            - Connections with other mentioned topics
            - Key statements and conclusions

            Text: {chunk}"""
            
            try:
                response = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=MAX_TOKENS
                )
                
                intermediate_summaries.append(response.choices[0].message.content)
                
            except Exception as e:
                st.error(f"Error with Groq API during intermediate summarization: {str(e)}")
                return None
        
        # Combine intermediate summaries for final processing
        combined_summary = "\n\n=== Next Section ===\n\n".join(intermediate_summaries)
        
        # Generate final comprehensive summary
        final_system_prompt = f"""You are an expert in creating comprehensive summaries. 
        Create a coherent, well-structured complete summary in {language_code} from the 
        provided intermediate summaries. Connect the information logically and establish 
        important relationships."""
        
        final_user_prompt = f"""Create a final, comprehensive summary from the following 
        intermediate summaries. The summary should:
        - Include all important topics and arguments
        - Establish logical connections between topics
        - Have a clear structure
        - Highlight key statements and most important insights
        
        Intermediate summaries:
        {combined_summary}"""
        
        try:
            final_response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_user_prompt}
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS
            )
            
            return final_response.choices[0].message.content
        except Exception as e:
            st.error(f"Error with Groq API during final summarization: {str(e)}")
            return None

    @staticmethod
    def get_youtube_title(video_url: str) -> str:
        """Fetch YouTube video title using requests and BeautifulSoup"""
        try:
            response = requests.get(video_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find('title')
            
            if title_tag:
                return title_tag.text.replace('- YouTube', '').strip()
            else:
                return "Untitled Video"
        except Exception as e:
            st.warning(f"Could not fetch video title: {str(e)}")
            return "Untitled Video"

    def save_chat(self, content: str, video_url: str) -> bool:
        """Save chat content to file with improved error handling"""
        try:
            # Generate filename from video title
            title = self.get_youtube_title(video_url)
            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', title)
            safe_filename = safe_filename[:100]  # Limit filename length
            
            file_path = CHATS_DIR / f"{safe_filename}.txt"
            
            # Handle duplicate filenames
            counter = 1
            original_path = file_path
            while file_path.exists():
                stem = original_path.stem
                file_path = CHATS_DIR / f"{stem}_{counter}.txt"
                counter += 1
            
            # Write content to file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            
            st.success(f"Chat saved as: {file_path.name}")
            return True
            
        except Exception as e:
            st.error(f"Error saving chat: {str(e)}")
            return False

    @staticmethod
    def get_chat_list() -> List[str]:
        """Get list of saved chat files"""
        if not CHATS_DIR.exists():
            return []
        
        chat_files = [
            file.stem for file in CHATS_DIR.glob("*.txt")
            if file.is_file()
        ]
        return sorted(chat_files)

    @staticmethod
    def display_chat(file_name: str):
        """Display saved chat content"""
        try:
            file_path = CHATS_DIR / f"{file_name}.txt"
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                st.header(f"{file_name.replace('_', ' ').title()}")
                st.markdown(content)
        except FileNotFoundError:
            st.error(f"The chat file '{file_name}.txt' does not exist.")
        except Exception as e:
            st.error(f"Error reading chat file: {str(e)}")


def update_env():
    """Handle environment variable updates through UI"""
    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    if st.button("Add Credentials"):
        st.session_state.show_form = True

    if st.session_state.show_form:
        with st.expander("Enter your credentials", expanded=True):
            email = st.text_input("Enter your YouTube email")
            password = st.text_input("Enter your YouTube password", type="password")
            api = st.text_input("Enter your Groq API key", type="password")

            if st.button("Submit"):
                if email and password and api:
                    # Ensure .env file exists
                    Path(ENV_FILE).touch()
                    
                    set_key(ENV_FILE, "YOUTUBE_EMAIL", email)
                    set_key(ENV_FILE, "YOUTUBE_PASSWORD", password)
                    set_key(ENV_FILE, "GROQ_API_KEY", api)
                    
                    st.success("✅ Credentials saved successfully.")
                    st.session_state.show_form = False
                else:
                    st.error("⚠️ Please fill in all fields.")


def main():
    """Main application function"""
    # Initialize the summarizer
    summarizer = YouTubeSummarizer()
    
    # Setup sidebar with chat history
    chat_list = summarizer.get_chat_list()
    chat_options = ['New Chat'] + chat_list
    
    with st.sidebar:
        selected = option_menu(
            "Chat History", 
            chat_options,
            icons=['plus'] + ['chat'] * len(chat_list), 
            menu_icon=""
        )

    # Main interface
    st.title('📺 YouTube Video Summarizer')
    st.markdown("""
    This tool creates comprehensive summaries of YouTube videos using advanced AI technology.
    It works with videos that have transcripts available!
    """)

    # Credentials management
    update_env()
    
    # Input controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_url = st.text_input('🔗 Enter YouTube video URL:')
    
    with col2:
        languages = summarizer.get_available_languages()
        target_language = st.selectbox(
            '🌍 Summary Language:',
            options=list(languages.keys()),
            index=0
        )
        target_language_code = languages[target_language]

    # Show available transcripts when URL is provided
    if video_url:
        available_transcripts = summarizer.get_available_transcripts(video_url)
        if available_transcripts:
            col3, col4 = st.columns([1, 1])
            
            with col3:
                # Create options for transcript language selection
                transcript_options = []
                transcript_codes = []
                
                for transcript in available_transcripts:
                    label = f"{transcript['language']} ({transcript['language_code']})"
                    if transcript['is_generated']:
                        label += " [Auto]"
                    transcript_options.append(label)
                    transcript_codes.append(transcript['language_code'])
                
                # Add "Auto-select" option
                transcript_options.insert(0, "🤖 Auto-select best match")
                transcript_codes.insert(0, "auto")
                
                selected_transcript_idx = st.selectbox(
                    '📝 Select Transcript:',
                    range(len(transcript_options)),
                    format_func=lambda x: transcript_options[x],
                    index=0
                )
                
                selected_transcript_code = transcript_codes[selected_transcript_idx]
                
            with col4:
                mode = st.selectbox(
                    '🎙️ Select Mode:',
                    options=['Video', 'Podcast'],
                    index=0
                ).lower()
        else:
            # If no URL provided or transcripts not available, show mode selection
            col3, col4 = st.columns([1, 1])
            with col3:
                st.info("Enter a YouTube URL to see available transcripts")
            with col4:
                mode = st.selectbox(
                    '🎙️ Select Mode:',
                    options=['Video', 'Podcast'],
                    index=0
                ).lower()
            selected_transcript_code = "auto"

    # Generate summary
    if st.button('Generate Summary'):
        if video_url:
            try:
                with st.spinner('Processing...'):
                    progress = st.progress(0)
                    status_text = st.empty()

                    # Fetch transcript with language preference
                    status_text.text('📥 Fetching video transcript...')
                    progress.progress(25)
                    
                    # Determine preferred transcript language
                    if selected_transcript_code == "auto":
                        # Auto-select: prefer target language, fallback to English
                        preferred_lang = target_language_code
                    else:
                        preferred_lang = selected_transcript_code
                    
                    transcript, transcript_language = summarizer.get_transcript(video_url, preferred_lang)
                    
                    if transcript:
                        # Generate summary
                        status_text.text(f'🤖 Generating {target_language} summary...')
                        progress.progress(75)

                        summary = summarizer.summarize_content(
                            transcript, 
                            target_language_code,
                            model_name=DEFAULT_MODEL,
                            mode=mode
                        )

                        if summary:
                            status_text.text('✨ Summary Ready!')
                            progress.progress(100)
                            
                            # Show transcript language used
                            if transcript_language:
                                st.info(f"📝 Transcript language used: {transcript_language}")
                            
                            # Display summary
                            st.markdown(summary)
                            
                            # Save chat
                            if summarizer.save_chat(summary, video_url):
                                st.rerun()
                        else:
                            st.error("Failed to generate summary.")
                    else:
                        st.error("Could not fetch transcript from the video.")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning('Please enter a valid YouTube link.')

    # Display selected chat
    if selected != "New Chat":
        summarizer.display_chat(selected)


if __name__ == "__main__":
    main()