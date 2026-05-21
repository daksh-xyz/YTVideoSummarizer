import os
import re
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import streamlit as st
from openai import OpenAI
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, set_key
from streamlit_option_menu import option_menu
from bs4 import BeautifulSoup


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CONSTANTS ---------------- #

CHATS_DIR = Path("./chats")
ENV_FILE = ".env"

DEFAULT_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200
MAX_TOKENS = 1024


# ---------------- MAIN CLASS ---------------- #

class YouTubeSummarizer:

    def __init__(self):
        self.groq_client = None
        self.transcript_api = YouTubeTranscriptApi()

        self._initialize_client()
        self._ensure_chats_folder()

    # ---------- SETUP ---------- #

    def _initialize_client(self):

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

        env_path = Path(__file__).parent / ENV_FILE

        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

        return api_key

    def _ensure_chats_folder(self):
        CHATS_DIR.mkdir(exist_ok=True)

    # ---------- YOUTUBE ---------- #

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:

        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$"
        ]

        youtube_url = youtube_url.strip()

        for pattern in patterns:
            match = re.search(pattern, youtube_url)

            if match:
                return match.group(1)

        raise ValueError("Could not extract video ID")

    def get_transcript(
        self,
        youtube_url: str,
        preferred_language: str = "en"
    ) -> Tuple[Optional[str], Optional[str]]:

        try:
            video_id = self.extract_video_id(youtube_url)

            fetched_transcript = self.transcript_api.fetch(
                video_id,
                languages=[preferred_language, "en"],
                preserve_formatting=False
            )

            if not fetched_transcript:
                st.error("No transcript found.")
                return None, None

            full_transcript = " ".join(
                [snippet.text for snippet in fetched_transcript]
            )

            if not full_transcript.strip():
                st.error("Transcript is empty.")
                return None, None

            st.success(
                f"Fetched transcript in {fetched_transcript.language}"
            )

            return (
                full_transcript,
                fetched_transcript.language_code
            )

        except (NoTranscriptFound, TranscriptsDisabled):
            st.error(
                "No captions available for this video."
            )
            return None, None

        except Exception as e:
            st.error(
                f"Error fetching transcript: {repr(e)}"
            )
            return None, None

    def get_available_transcripts(
        self,
        youtube_url: str
    ) -> List[Dict[str, str]]:

        try:
            video_id = self.extract_video_id(youtube_url)

            transcript_list = self.transcript_api.list(video_id)

            transcripts_info = []

            for transcript in transcript_list:

                transcripts_info.append({
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "is_translatable": transcript.is_translatable
                })

            return transcripts_info

        except Exception:
            return []

    # ---------- LANGUAGES ---------- #

    @staticmethod
    def get_available_languages() -> Dict[str, str]:

        return {
            "English": "en",
            "Deutsch": "de",
            "Italiano": "it",
            "Español": "es",
            "Français": "fr",
            "Nederlands": "nl",
            "Polski": "pl",
            "日本語": "ja",
            "中文": "zh",
            "Русский": "ru"
        }

    # ---------- SUMMARIZATION ---------- #

    def summarize_content(
        self,
        transcript: str,
        language_code: str,
        model_name: str = DEFAULT_MODEL
    ) -> Optional[str]:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

        text_chunks = text_splitter.split_text(transcript)

        # SMALL TRANSCRIPTS -> DIRECT SUMMARY

        if len(text_chunks) == 1:

            try:

                response = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                            You are an expert YouTube video summarizer.
                            Create a structured and detailed summary in {language_code}.
                            """
                        },
                        {
                            "role": "user",
                            "content": f"""
                            Summarize this transcript thoroughly.

                            Transcript:
                            {transcript}
                            """
                        }
                    ],
                    temperature=0.5,
                    max_tokens=MAX_TOKENS
                )

                return response.choices[0].message.content

            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")
                return None

        # LARGE TRANSCRIPTS -> MAP REDUCE

        intermediate_summaries = []

        for i, chunk in enumerate(text_chunks):

            try:

                response = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                            You are an expert summarizer.
                            Summarize this section in {language_code}.
                            """
                        },
                        {
                            "role": "user",
                            "content": chunk
                        }
                    ],
                    temperature=0.5,
                    max_tokens=MAX_TOKENS
                )

                intermediate_summaries.append(
                    response.choices[0].message.content
                )

            except Exception as e:
                st.error(
                    f"Error during chunk summarization: {str(e)}"
                )
                return None

        combined_summary = "\n\n".join(intermediate_summaries)

        try:

            final_response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        Create a final polished comprehensive summary
                        in {language_code}.
                        """
                    },
                    {
                        "role": "user",
                        "content": combined_summary
                    }
                ],
                temperature=0.5,
                max_tokens=MAX_TOKENS
            )

            return final_response.choices[0].message.content

        except Exception as e:
            st.error(
                f"Error during final summarization: {str(e)}"
            )
            return None

    # ---------- FILE HANDLING ---------- #

    @staticmethod
    def get_youtube_title(video_url: str) -> str:

        try:

            response = requests.get(video_url, timeout=10)

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            title_tag = soup.find("title")

            if title_tag:
                return (
                    title_tag.text
                    .replace("- YouTube", "")
                    .strip()
                )

            return "Untitled Video"

        except Exception:
            return "Untitled Video"

    def save_chat(
        self,
        content: str,
        video_url: str
    ) -> bool:

        try:

            title = self.get_youtube_title(video_url)

            safe_filename = re.sub(
                r'[<>:"/\\|?*]',
                "_",
                title
            )

            safe_filename = safe_filename[:100]

            file_path = CHATS_DIR / f"{safe_filename}.txt"

            counter = 1
            original_path = file_path

            while file_path.exists():

                stem = original_path.stem

                file_path = (
                    CHATS_DIR /
                    f"{stem}_{counter}.txt"
                )

                counter += 1

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(content)

            st.success(f"Saved chat: {file_path.name}")

            return True

        except Exception as e:

            st.error(f"Error saving chat: {str(e)}")

            return False

    @staticmethod
    def get_chat_list() -> List[str]:

        if not CHATS_DIR.exists():
            return []

        return sorted([
            file.stem
            for file in CHATS_DIR.glob("*.txt")
            if file.is_file()
        ])

    @staticmethod
    def display_chat(file_name: str):

        try:

            file_path = CHATS_DIR / f"{file_name}.txt"

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

                st.header(
                    file_name.replace("_", " ").title()
                )

                st.markdown(content)

        except FileNotFoundError:
            st.error("Chat file not found.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


# ---------------- ENV UI ---------------- #

def update_env():

    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    if st.button("Add Credentials"):
        st.session_state.show_form = True

    if st.session_state.show_form:

        with st.expander(
            "Enter your credentials",
            expanded=True
        ):

            api = st.text_input(
                "Enter your Groq API key",
                type="password"
            )

            if st.button("Submit"):

                if api:

                    Path(ENV_FILE).touch()

                    set_key(
                        ENV_FILE,
                        "GROQ_API_KEY",
                        api
                    )

                    st.success(
                        "Credentials saved successfully."
                    )

                    st.session_state.show_form = False

                else:
                    st.error("Please enter API key.")


# ---------------- MAIN ---------------- #

def main():

    summarizer = YouTubeSummarizer()

    chat_list = summarizer.get_chat_list()

    chat_options = ["New Chat"] + chat_list

    with st.sidebar:

        selected = option_menu(
            "Chat History",
            chat_options,
            icons=["plus"] + ["chat"] * len(chat_list),
            menu_icon=""
        )

    if selected != "New Chat":

        summarizer.display_chat(selected)

        return

    st.title("📺 YouTube Video Summarizer")

    st.markdown("""
    Summarize YouTube videos using AI.
    """)

    update_env()

    youtube_url = st.text_input(
        "Enter YouTube Video URL"
    )

    languages = summarizer.get_available_languages()

    selected_language = st.selectbox(
        "Select Transcript Language",
        options=list(languages.keys())
    )

    if st.button("Generate Summary"):

        if not youtube_url.strip():
            st.warning("Please enter a YouTube URL.")
            return

        with st.spinner("Fetching transcript..."):

            transcript, language_code = (
                summarizer.get_transcript(
                    youtube_url,
                    languages[selected_language]
                )
            )

        if transcript:

            with st.spinner("Generating summary..."):

                summary = summarizer.summarize_content(
                    transcript,
                    language_code
                )

            if summary:

                st.markdown(summary)

                summarizer.save_chat(
                    summary,
                    youtube_url
                )


if __name__ == "__main__":
    main()
