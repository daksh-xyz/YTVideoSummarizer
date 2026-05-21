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
    TranscriptsDisabled,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, set_key
from streamlit_option_menu import option_menu
from bs4 import BeautifulSoup


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

CHATS_DIR = Path("./chats")
ENV_FILE = ".env"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200
MAX_TOKENS = 1024

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ── Sidebar style ─────────────────────────────────────────────────────────────

# Truncate long chat names in the sidebar menu with an ellipsis.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .nav-link span {
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 170px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Filename helpers ──────────────────────────────────────────────────────────

def _make_safe_stem(title: str, max_len: int = 100) -> str:
    stem = title.strip()
    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = re.sub(r"[\s_]+", "_", stem)
    stem = stem.strip("._")
    stem = re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", stem)
    stem = stem.strip("._")
    stem = stem or "Untitled_Video"
    return stem[:max_len]


def _unique_path(stem: str) -> Path:
    candidate = CHATS_DIR / f"{stem}.txt"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = CHATS_DIR / f"{stem}_{counter}.txt"
        if not candidate.exists():
            return candidate
        counter += 1


# ── Main class ────────────────────────────────────────────────────────────────

class YouTubeSummarizer:

    def __init__(self):
        self.groq_client: Optional[OpenAI] = None
        self.transcript_api = YouTubeTranscriptApi()
        self._initialize_client()
        self._ensure_chats_folder()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _initialize_client(self) -> None:
        try:
            api_key = self._load_environment()
            self.groq_client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except Exception as e:
            st.error(f"Error initialising API client: {e}")
            st.stop()

    def _load_environment(self) -> str:
        env_path = Path(__file__).parent / ENV_FILE
        if env_path.exists():
            load_dotenv(env_path)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or .env file.")
        return api_key

    @staticmethod
    def _ensure_chats_folder() -> None:
        CHATS_DIR.mkdir(exist_ok=True)

    # ── YouTube ────────────────────────────────────────────────────────────

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, youtube_url.strip())
            if match:
                return match.group(1)
        raise ValueError("Could not extract a valid YouTube video ID from the URL.")

    def get_transcript(
        self,
        youtube_url: str,
        preferred_language: str = "en",
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            video_id = self.extract_video_id(youtube_url)
            fetched = self.transcript_api.fetch(
                video_id,
                languages=[preferred_language, "en"],
                preserve_formatting=False,
            )
            if not fetched:
                st.error("No transcript found for this video.")
                return None, None
            full_text = " ".join(snippet.text for snippet in fetched).strip()
            if not full_text:
                st.error("Transcript appears to be empty.")
                return None, None
            st.success(f"Transcript fetched in: {fetched.language}")
            return full_text, fetched.language_code
        except (NoTranscriptFound, TranscriptsDisabled):
            st.error("No captions are available for this video.")
            return None, None
        except Exception as e:
            st.error(f"Error fetching transcript: {e!r}")
            return None, None

    def get_available_transcripts(self, youtube_url: str) -> List[Dict[str, str]]:
        try:
            video_id = self.extract_video_id(youtube_url)
            return [
                {
                    "language": t.language,
                    "language_code": t.language_code,
                    "is_generated": t.is_generated,
                    "is_translatable": t.is_translatable,
                }
                for t in self.transcript_api.list(video_id)
            ]
        except Exception:
            return []

    # ── Languages ──────────────────────────────────────────────────────────

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
            "Русский": "ru",
        }

    # ── Summarisation prompts ──────────────────────────────────────────────

    @staticmethod
    def _system_prompt(language_code: str, role: str = "full") -> str:
        lang_instruction = (
            f"Write your entire response in the language with ISO code '{language_code}'."
            if language_code != "en"
            else "Write your entire response in English."
        )
        if role == "chunk":
            return f"""\
You are a precise summariser working on one section of a YouTube video transcript.

Rules:
- Capture every key idea, fact, argument, claim, and example in this section.
- Use Markdown headings (## / ###) to organise material logically.
- Use bullet points for lists or steps; short prose for explanations.
- Preserve specific names, numbers, dates, and technical terms verbatim.
- Never write meta-commentary ("Here is a summary of…", "This section covers…").
- Never mention transcripts, captions, or the summarisation process.

{lang_instruction}"""
        if role == "reduce":
            return f"""\
You are a senior editor merging several section-level summaries of a YouTube video \
into one cohesive, publication-ready document.

Rules:
- Begin with a **Overview** paragraph (2–4 sentences) a new reader can understand instantly.
- Merge all input summaries into clearly structured sections with ## / ### headings.
- Remove repetition, but retain every unique fact, insight, and example.
- End with a **Key Takeaways** section containing 3–7 concise bullet points.
- Maintain a neutral, informative tone throughout.
- Never write meta-commentary or reference the summarisation process.

{lang_instruction}"""
        return f"""\
You are an expert at turning YouTube videos into clear, well-structured written summaries.

Rules:
- Begin with a short **Overview** paragraph (2–4 sentences) capturing the video's \
purpose, topic, and intended audience.
- Organise the rest with ## / ### Markdown headings that mirror the video's natural structure.
- Use bullet points for enumerated items, steps, or lists; prose for context and explanation.
- Preserve specific names, numbers, dates, technical terms, and any claims made in the video.
- End with a **Key Takeaways** section: 3–7 punchy bullet points the viewer should remember.
- Never write any meta-commentary ("Here is a summary…", "Based on the transcript…").
- Never mention transcripts, captions, or the summarisation process.

{lang_instruction}"""

    # ── Summarisation ──────────────────────────────────────────────────────

    def summarize_content(
        self,
        transcript: str,
        language_code: str,
        model_name: str = DEFAULT_MODEL,
    ) -> Optional[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_text(transcript)

        if len(chunks) == 1:
            try:
                resp = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self._system_prompt(language_code, "full")},
                        {"role": "user",   "content": transcript},
                    ],
                    temperature=0.4,
                    max_tokens=MAX_TOKENS,
                )
                return resp.choices[0].message.content
            except Exception as e:
                st.error(f"Groq API error: {e}")
                return None

        intermediate: List[str] = []
        progress = st.progress(0.0, text="Summarising sections…")
        for i, chunk in enumerate(chunks):
            try:
                resp = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self._system_prompt(language_code, "chunk")},
                        {"role": "user",   "content": chunk},
                    ],
                    temperature=0.4,
                    max_tokens=MAX_TOKENS,
                )
                intermediate.append(resp.choices[0].message.content)
                progress.progress(
                    (i + 1) / len(chunks),
                    text=f"Section {i + 1} of {len(chunks)} complete…",
                )
            except Exception as e:
                st.error(f"Error summarising section {i + 1}: {e}")
                return None
        progress.empty()

        combined = "\n\n---\n\n".join(intermediate)
        try:
            final_resp = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt(language_code, "reduce")},
                    {"role": "user",   "content": combined},
                ],
                temperature=0.4,
                max_tokens=MAX_TOKENS,
            )
            return final_resp.choices[0].message.content
        except Exception as e:
            st.error(f"Error during final synthesis: {e}")
            return None

    # ── File handling ──────────────────────────────────────────────────────

    @staticmethod
    def get_youtube_title(video_url: str) -> str:
        try:
            resp = requests.get(video_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            tag = soup.find("title")
            if tag:
                return tag.text.replace("- YouTube", "").strip()
        except Exception:
            pass
        return "Untitled Video"

    def save_chat(self, content: str, video_url: str) -> Optional[str]:
        """Save summary to disk. Returns the file stem on success, None on failure."""
        try:
            raw_title = self.get_youtube_title(video_url)
            stem = _make_safe_stem(raw_title)
            file_path = _unique_path(stem)
            file_path.write_text(content, encoding="utf-8")
            return file_path.stem   # caller handles UI feedback
        except Exception as e:
            st.error(f"Error saving chat: {e}")
            return None

    @staticmethod
    def get_chat_list() -> List[str]:
        if not CHATS_DIR.exists():
            return []
        return sorted(f.stem for f in CHATS_DIR.glob("*.txt") if f.is_file())

    @staticmethod
    def display_chat(file_stem: str) -> None:
        file_path = CHATS_DIR / f"{file_stem}.txt"
        try:
            content = file_path.read_text(encoding="utf-8")
            display_title = file_stem.replace("_", " ").title()
            st.header(display_title)
            st.markdown(content)
        except FileNotFoundError:
            st.error("Chat file not found.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    @staticmethod
    def rename_chat(old_stem: str, new_name: str) -> Optional[str]:
        """
        Rename a chat file. Returns the new stem on success, None on failure.
        Handles collisions with _unique_path.
        """
        new_stem = _make_safe_stem(new_name)
        if not new_stem:
            st.error("Please enter a valid name.")
            return None
        old_path = CHATS_DIR / f"{old_stem}.txt"
        if not old_path.exists():
            st.error("Original file not found.")
            return None
        # Avoid unnecessary rename if names are the same
        if new_stem == old_stem:
            return old_stem
        new_path = _unique_path(new_stem)
        try:
            old_path.rename(new_path)
            return new_path.stem
        except Exception as e:
            st.error(f"Error renaming chat: {e}")
            return None

    @staticmethod
    def delete_chat(stem: str) -> bool:
        """Delete a chat file. Returns True on success."""
        file_path = CHATS_DIR / f"{stem}.txt"
        try:
            file_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            st.error(f"Error deleting chat: {e}")
            return False


# ── ENV UI ────────────────────────────────────────────────────────────────────

def update_env() -> None:
    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    if st.button("Add Credentials"):
        st.session_state.show_form = True

    if st.session_state.show_form:
        with st.expander("Enter your credentials", expanded=True):
            api = st.text_input("Groq API key", type="password")
            if st.button("Save"):
                if api:
                    Path(ENV_FILE).touch()
                    set_key(ENV_FILE, "GROQ_API_KEY", api)
                    os.environ["GROQ_API_KEY"] = api   # apply immediately, no restart needed
                    st.success("Credentials saved.")
                    st.session_state.show_form = False
                else:
                    st.error("Please enter a non-empty API key.")


# ── Chat management sidebar panel ─────────────────────────────────────────────

def render_chat_actions(summarizer: "YouTubeSummarizer", stem: str) -> None:
    """Render rename and delete controls in the sidebar for the selected chat."""
    st.sidebar.markdown("---")

    # ── Rename ──────────────────────────────────────────────────────────────
    with st.sidebar.expander("✏️ Rename"):
        display_name = stem.replace("_", " ").title()
        new_name = st.text_input(
            "New name",
            value=display_name,
            key="rename_input",
            label_visibility="collapsed",
        )
        if st.button("Rename", key="rename_btn", use_container_width=True):
            new_stem = summarizer.rename_chat(stem, new_name)
            if new_stem:
                st.session_state.selected_chat = new_stem
                st.rerun()

    # ── Delete ──────────────────────────────────────────────────────────────
    with st.sidebar.expander("🗑️ Delete"):
        st.warning(f'Delete **"{stem.replace("_", " ").title()}"**? This cannot be undone.')
        if st.button("Delete", key="delete_btn", type="primary", use_container_width=True):
            if summarizer.delete_chat(stem):
                st.session_state.selected_chat = None
                st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    summarizer = YouTubeSummarizer()

    # selected_chat persists across reruns so a newly saved chat is shown immediately
    if "selected_chat" not in st.session_state:
        st.session_state.selected_chat = None

    chat_list = summarizer.get_chat_list()
    chat_options = ["New Chat"] + chat_list

    # Determine the default index for option_menu based on session state
    if st.session_state.selected_chat and st.session_state.selected_chat in chat_list:
        default_idx = chat_options.index(st.session_state.selected_chat)
    else:
        default_idx = 0

    with st.sidebar:
        selected = option_menu(
            "Chat History",
            chat_options,
            icons=["plus"] + ["chat-text"] * len(chat_list),
            menu_icon="collection",
            default_index=default_idx,
            key="sidebar_menu",
        )

    # Keep session state in sync with whatever the user clicked
    st.session_state.selected_chat = selected if selected != "New Chat" else None

    # ── Viewing an existing chat ───────────────────────────────────────────
    if selected != "New Chat":
        summarizer.display_chat(selected)
        render_chat_actions(summarizer, selected)
        return

    # ── New chat page ──────────────────────────────────────────────────────
    st.title("📺 YouTube Video Summarizer")
    st.markdown("Summarise any YouTube video using AI — paste a URL and go.")

    update_env()

    youtube_url = st.text_input("YouTube video URL")

    languages = summarizer.get_available_languages()
    selected_language = st.selectbox(
        "Transcript language",
        options=list(languages.keys()),
    )

    if st.button("Generate Summary", type="primary", use_container_width=True):
        if not youtube_url.strip():
            st.warning("Please enter a YouTube URL.")
            return

        with st.spinner("Fetching transcript…"):
            transcript, language_code = summarizer.get_transcript(
                youtube_url, languages[selected_language]
            )

        if transcript:
            with st.spinner("Generating summary…"):
                summary = summarizer.summarize_content(transcript, language_code)

            if summary:
                st.markdown(summary)

                saved_stem = summarizer.save_chat(summary, youtube_url)
                if saved_stem:
                    # Switch to the new chat so it appears selected in the sidebar
                    st.session_state.selected_chat = saved_stem
                    st.rerun()


if __name__ == "__main__":
    main()
