"""
SyncIn — Streamlit version of the multimodal emotion music player.
"""
import os
import sys
import json
import tempfile
import urllib.parse
import urllib.request
import shutil
import subprocess

import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.emotion_detection.facial_emotion import FacialEmotionDetector
from src.emotion_detection.audio_emotion import AudioEmotionDetector
from src.emotion_detection.text_emotion import TextEmotionDetector
from src.fusion.multimodal_fusion import MultimodalFusion
from src.music_analysis.music_emotion_recognition import MusicEmotionAnalyzer
from src.recommendation.recommendation_engine import MusicRecommendationEngine


EMOJI = {"angry": "😠", "happy": "😊", "neutral": "😐", "sad": "😢"}


@st.cache_resource(show_spinner="Loading models…")
def get_components():
    facial = FacialEmotionDetector()
    audio = AudioEmotionDetector()
    text = TextEmotionDetector()
    fusion = MultimodalFusion(fusion_method="attention")
    analyzer = MusicEmotionAnalyzer()
    recommender = MusicRecommendationEngine(analyzer)
    return facial, audio, text, fusion, recommender


def fetch_preview(query):
    q = urllib.parse.quote(query)
    try:
        with urllib.request.urlopen(
            f"https://itunes.apple.com/search?term={q}&media=music&limit=1", timeout=8
        ) as r:
            data = json.loads(r.read())
        if data.get("results"):
            t = data["results"][0]
            if t.get("previewUrl"):
                return t["previewUrl"], f'{t["artistName"]} - {t["trackName"]}', t.get("artworkUrl100")
    except Exception:
        pass

    try:
        with urllib.request.urlopen(
            f"https://api.deezer.com/search?q={q}&limit=1", timeout=8
        ) as r:
            data = json.loads(r.read())
        if data.get("data"):
            t = data["data"][0]
            if t.get("preview"):
                return t["preview"], f'{t["artist"]["name"]} - {t["title"]}', t.get("album", {}).get("cover_medium")
    except Exception:
        pass

    return None, None, None


def download_and_convert(url):
    path = urllib.parse.urlparse(url).path.lower()
    is_aac = path.endswith(".m4a") or path.endswith(".aac")
    suffix = ".m4a" if is_aac else ".mp3"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with urllib.request.urlopen(url, timeout=15) as r:
        tmp.write(r.read())
    tmp.close()

    if not is_aac:
        return tmp.name, "audio/mp3"

    if not shutil.which("ffmpeg"):
        return tmp.name, "audio/mp4"

    mp3_path = tmp.name.replace(".m4a", ".mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
             "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
            check=True,
        )
        os.unlink(tmp.name)
        return mp3_path, "audio/mp3"
    except subprocess.CalledProcessError:
        return tmp.name, "audio/mp4"


def detect_facial_from_image(facial_detector, image_bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(image_bytes)
    tmp.close()
    try:
        return facial_detector.get_emotion_probabilities(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def detect_audio_from_bytes(audio_detector, audio_bytes):
    import soundfile as sf
    import io
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != audio_detector.sample_rate:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=audio_detector.sample_rate)
    return audio_detector.get_emotion_probabilities(data.astype(np.float32))


def init_state():
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("current_idx", 0)
    st.session_state.setdefault("emotion", None)
    st.session_state.setdefault("emotion_probs", None)


def render_player(track_query):
    url, resolved, art = fetch_preview(track_query)
    if not url:
        st.warning(f"No preview found for: {track_query}")
        return

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if art:
            st.image(art)
    with col_b:
        st.markdown(f"### {resolved}")
        if st.session_state.emotion:
            st.markdown(f"**Mood:** {EMOJI.get(st.session_state.emotion, '')} {st.session_state.emotion.title()}")

    try:
        path, mime = download_and_convert(url)
        with open(path, "rb") as f:
            st.audio(f.read(), format=mime)
        os.unlink(path)
    except Exception as e:
        st.error(f"Playback failed: {e}")


def main():
    st.set_page_config(page_title="SyncIn", page_icon="🎵", layout="centered")
    init_state()

    st.title("🎵 SyncIn")
    st.caption("Multimodal emotion-aware music player")

    facial, audio, text, fusion, recommender = get_components()

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "What do you want to play?",
            ["Emotion-recommended", "Search a song", "Manual playlist"],
            label_visibility="collapsed",
        )
        st.divider()
        strategy = st.radio("Recommendation strategy", ["Mood matching", "Mood regulation"])
        recommender.set_strategy("mood_matching" if strategy == "Mood matching" else "mood_regulation")

    st.subheader("1. Detect your emotion")
    st.caption("All inputs optional — provide what you have.")

    img = st.camera_input("Take a face snapshot")
    voice = st.audio_input("Record your voice (optional)")
    typed = st.text_input("How are you feeling? (optional)", placeholder="e.g. excited about today")

    if st.button("🔍 Detect emotion", type="primary"):
        facial_probs = audio_probs = text_probs = None

        with st.spinner("Analyzing…"):
            if img is not None:
                try:
                    facial_probs = detect_facial_from_image(facial, img.getvalue())
                except Exception as e:
                    st.warning(f"Facial detection failed: {e}")

            if voice is not None:
                try:
                    audio_probs = detect_audio_from_bytes(audio, voice.getvalue())
                except Exception as e:
                    st.warning(f"Audio detection failed: {e}")

            if typed.strip():
                try:
                    text_probs = text.get_emotion_probabilities(typed.strip())
                except Exception as e:
                    st.warning(f"Text detection failed: {e}")

            if not any([facial_probs, audio_probs, text_probs]):
                st.error("Provide at least one input (camera, voice, or text).")
            else:
                fused = fusion.fuse_emotions(facial_probs, audio_probs, text_probs)
                emotion = max(fused, key=fused.get)
                st.session_state.emotion = emotion
                st.session_state.emotion_probs = fused

    if st.session_state.emotion_probs:
        st.divider()
        e = st.session_state.emotion
        st.success(f"Detected emotion: **{EMOJI.get(e, '')} {e.title()}** ({st.session_state.emotion_probs[e]*100:.1f}%)")
        with st.expander("Probability breakdown"):
            for k, v in sorted(st.session_state.emotion_probs.items(), key=lambda x: -x[1]):
                st.write(f"- {EMOJI.get(k, '')} {k.title()}: {v*100:.1f}%")

    st.divider()
    st.subheader("2. Build your queue")

    if mode == "Emotion-recommended":
        if st.session_state.emotion:
            if st.button("🎯 Get recommendation"):
                song = recommender.recommend_song(st.session_state.emotion)
                if song:
                    st.session_state.queue = [song]
                    st.session_state.current_idx = 0
                else:
                    st.warning("No song available for this emotion")
        else:
            st.info("Detect an emotion first.")

    elif mode == "Search a song":
        q = st.text_input("Song name", placeholder="Artist - Title")
        if st.button("➕ Add to queue") and q.strip():
            st.session_state.queue = [q.strip()]
            st.session_state.current_idx = 0

    else:
        raw = st.text_area(
            "Song names, one per line",
            placeholder="Arcade Fire - Wake Up\nColdplay - Yellow\nDaft Punk - Get Lucky",
            height=160,
        )
        if st.button("➕ Load queue"):
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if lines:
                st.session_state.queue = lines
                st.session_state.current_idx = 0
            else:
                st.warning("Add at least one song.")

    if st.session_state.queue:
        st.divider()
        st.subheader("3. Now playing")

        idx = st.session_state.current_idx
        total = len(st.session_state.queue)
        st.caption(f"Track {idx + 1} / {total}")

        render_player(st.session_state.queue[idx])

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⏮️ Prev", disabled=idx == 0):
                st.session_state.current_idx -= 1
                st.rerun()
        with c2:
            if st.button("⏭️ Next", disabled=idx >= total - 1):
                st.session_state.current_idx += 1
                st.rerun()
        with c3:
            if st.button("🗑️ Clear queue"):
                st.session_state.queue = []
                st.session_state.current_idx = 0
                st.rerun()

        with st.expander(f"Queue ({total} tracks)"):
            for i, q in enumerate(st.session_state.queue):
                marker = "▶️" if i == idx else "  "
                st.write(f"{marker} {i+1}. {q}")


if __name__ == "__main__":
    main()
