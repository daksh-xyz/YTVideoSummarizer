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


# ── Sidebar styles ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Chat row buttons ── */

    /* The chat-name button: left-aligned, full width, no background */
    [data-testid="stSidebar"] button[data-chat-name] {
        background: none;
        border: none;
        text-align: left;
        padding: 0.35rem 0.5rem;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border-radius: 6px;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] button[data-chat-name]:hover {
        background: rgba(255,255,255,0.07);
    }

    /* The ⋮ button: compact, no background */
    [data-testid="stSidebar"] button[data-dots] {
        background: none;
        border: none;
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
        font-size: 1.1rem;
        line-height: 1;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] button[data-dots]:hover {
        background: rgba(255,255,255,0.1);
    }

    /* Highlighted (selected) chat row */
    [data-testid="stSidebar"] div[data-selected-row] button[data-chat-name] {
        background: rgba(255,255,255,0.12);
        font-weight: 600;
    }

    /* Rename / delete action buttons inside the inline panel */
    [data-testid="stSidebar"] button[data-action="rename"],
    [data-testid="stSidebar"] button[data-action="delete"] {
        border-radius: 5px;
        font-size: 0.82rem;
        padding: 0.25rem 0.6rem;
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


def _display_name(stem: str) -> str:
    return stem.replace("_", " ").title()


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

    # ── Prompts ────────────────────────────────────────────────────────────

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
        try:
            raw_title = self.get_youtube_title(video_url)
            stem = _make_safe_stem(raw_title)
            file_path = _unique_path(stem)
            file_path.write_text(content, encoding="utf-8")
            return file_path.stem
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
            st.header(_display_name(file_stem))
            st.markdown(content)
        except FileNotFoundError:
            st.error("Chat file not found.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    @staticmethod
    def rename_chat(old_stem: str, new_name: str) -> Optional[str]:
        new_stem = _make_safe_stem(new_name)
        if not new_stem:
            st.error("Please enter a valid name.")
            return None
        if new_stem == old_stem:
            return old_stem
        old_path = CHATS_DIR / f"{old_stem}.txt"
        if not old_path.exists():
            st.error("Original file not found.")
            return None
        new_path = _unique_path(new_stem)
        try:
            old_path.rename(new_path)
            return new_path.stem
        except Exception as e:
            st.error(f"Error renaming chat: {e}")
            return None

    @staticmethod
    def delete_chat(stem: str) -> bool:
        try:
            (CHATS_DIR / f"{stem}.txt").unlink(missing_ok=True)
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
                    os.environ["GROQ_API_KEY"] = api
                    st.success("Credentials saved.")
                    st.session_state.show_form = False
                else:
                    st.error("Please enter a non-empty API key.")


# ── Sidebar chat list ─────────────────────────────────────────────────────────

def render_sidebar(summarizer: "YouTubeSummarizer") -> str:
    """
    Render the full sidebar and return the currently selected chat stem,
    or "New Chat".

    Each chat row has:
      [chat name button (fills width)]  [⋮ button]

    Clicking ⋮ toggles an inline panel beneath that row with Rename and Delete.
    """
    # Session-state keys used here
    # selected_chat : str | None   — current open chat stem (None = New Chat)
    # dots_open     : str | None   — stem whose ⋮ menu is expanded
    # rename_mode   : str | None   — stem currently being renamed

    ss = st.session_state
    ss.setdefault("selected_chat", None)
    ss.setdefault("dots_open", None)
    ss.setdefault("rename_mode", None)

    chat_list = summarizer.get_chat_list()

    with st.sidebar:
        st.markdown("## 📺 Chats")

        # ── New Chat button ────────────────────────────────────────────────
        if st.button("＋  New Chat", use_container_width=True, key="btn_new_chat"):
            ss.selected_chat = None
            ss.dots_open = None
            ss.rename_mode = None

        st.markdown("---")

        # ── One row per saved chat ─────────────────────────────────────────
        for stem in chat_list:
            is_selected = ss.selected_chat == stem
            is_dots_open = ss.dots_open == stem
            is_renaming = ss.rename_mode == stem

            # Highlight selected row with a subtle background
            row_style = (
                "background:rgba(255,255,255,0.10);border-radius:8px;padding:2px 0;"
                if is_selected
                else "border-radius:8px;padding:2px 0;"
            )
            with st.container():
                st.markdown(f'<div style="{row_style}">', unsafe_allow_html=True)
                col_name, col_dots = st.columns([11, 1])

                # Chat name button
                label = _display_name(stem)
                # Truncate display label so it never wraps
                short_label = label if len(label) <= 22 else label[:20] + "…"
                with col_name:
                    if st.button(
                        short_label,
                        key=f"chat_btn_{stem}",
                        use_container_width=True,
                        help=label,           # full name on hover
                    ):
                        ss.selected_chat = stem
                        ss.dots_open = None
                        ss.rename_mode = None

                # ⋮ button
                with col_dots:
                    if st.button("⋮", key=f"dots_{stem}", help="Rename or delete"):
                        ss.dots_open = stem if not is_dots_open else None
                        ss.rename_mode = None   # close any open rename input

                st.markdown("</div>", unsafe_allow_html=True)

            # ── Inline action panel (shown below the row when ⋮ is open) ──
            if is_dots_open:
                with st.container():
                    st.markdown(
                        '<div style="margin-left:8px;margin-bottom:6px;">',
                        unsafe_allow_html=True,
                    )

                    action_col1, action_col2 = st.columns(2)

                    with action_col1:
                        if st.button(
                            "✏️ Rename",
                            key=f"action_rename_{stem}",
                            use_container_width=True,
                        ):
                            ss.rename_mode = stem if not is_renaming else None

                    with action_col2:
                        if st.button(
                            "🗑️ Delete",
                            key=f"action_delete_{stem}",
                            use_container_width=True,
                            type="primary",
                        ):
                            summarizer.delete_chat(stem)
                            if ss.selected_chat == stem:
                                ss.selected_chat = None
                            ss.dots_open = None
                            ss.rename_mode = None
                            st.rerun()
 
                    # Rename input (only when rename was clicked)
                    if is_renaming:
                        new_name = st.text_input(
                            "New name",
                            value=_display_name(stem),
                            key=f"rename_input_{stem}",
                            label_visibility="collapsed",
                        )
                        if st.button(
                            "Save name",
                            key=f"rename_save_{stem}",
                            use_container_width=True,
                        ):
                            new_stem = summarizer.rename_chat(stem, new_name)
                            if new_stem:
                                if ss.selected_chat == stem:
                                    ss.selected_chat = new_stem
                                ss.dots_open = None
                                ss.rename_mode = None
                                st.rerun()
 
                    st.markdown("</div>", unsafe_allow_html=True)
 
    return ss.selected_chat if ss.selected_chat else "New Chat"
    # ── Main ──────────────────────────────────────────────────────────────────────
 
def main() -> None:
    summarizer = YouTubeSummarizer()
 
    selected = render_sidebar(summarizer)
 
    # ── Viewing an existing chat ───────────────────────────────────────────
    if selected != "New Chat":
        summarizer.display_chat(selected)
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
                    st.session_state.selected_chat = saved_stem
                    st.rerun()
 
 
if __name__ == "__main__":
    main()
