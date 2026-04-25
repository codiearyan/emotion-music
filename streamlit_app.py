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

import html as html_lib

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

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
    st.session_state.setdefault("modality_probs", {})


PLAYER_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: transparent;
  }
  .card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 22px;
    color: #fff;
    display: flex;
    gap: 20px;
    align-items: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  }
  .art {
    width: 120px;
    height: 120px;
    border-radius: 12px;
    background-size: cover;
    background-position: center;
    background-color: #2a2a3e;
    flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  }
  .info { flex: 1; min-width: 0; }
  .title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mood {
    font-size: 13px;
    color: #a0a0c0;
    margin-bottom: 14px;
  }
  .progress-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }
  .time { font-size: 11px; color: #888; min-width: 36px; font-variant-numeric: tabular-nums; }
  .progress {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  .progress-fill {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 3px;
    width: 0%;
    transition: width 0.1s linear;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .btn-play {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border: none;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: transform 0.15s;
    box-shadow: 0 4px 14px rgba(99,102,241,0.5);
  }
  .btn-play:hover { transform: scale(1.08); }
  .btn-play:active { transform: scale(0.95); }
  .vol {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }
  .vol-icon { font-size: 14px; color: #888; }
  input[type=range] {
    -webkit-appearance: none;
    width: 70px;
    height: 4px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #fff;
    cursor: pointer;
  }
  @media (max-width: 480px) {
    .card { flex-direction: column; align-items: stretch; }
    .art { width: 100%; height: 200px; }
  }
</style>
</head>
<body>
  <div class="card">
    <div class="art" style="background-image: url('__ART__');"></div>
    <div class="info">
      <div class="title">__TITLE__</div>
      <div class="mood">__MOOD__</div>
      <div class="progress-row">
        <span class="time" id="cur">0:00</span>
        <div class="progress" id="prog"><div class="progress-fill" id="fill"></div></div>
        <span class="time" id="dur">0:00</span>
      </div>
      <div class="controls">
        <button class="btn-play" id="play">▶</button>
        <div class="vol">
          <span class="vol-icon">🔊</span>
          <input type="range" id="vol" min="0" max="100" value="80">
        </div>
      </div>
    </div>
    <audio id="audio" src="__SRC__" preload="auto"></audio>
  </div>
<script>
  const a = document.getElementById('audio');
  const playBtn = document.getElementById('play');
  const fill = document.getElementById('fill');
  const prog = document.getElementById('prog');
  const cur = document.getElementById('cur');
  const dur = document.getElementById('dur');
  const vol = document.getElementById('vol');

  const fmt = s => {
    if (!isFinite(s)) return '0:00';
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${r}`;
  };

  playBtn.onclick = () => {
    if (a.paused) { a.play(); playBtn.textContent = '⏸'; }
    else { a.pause(); playBtn.textContent = '▶'; }
  };
  a.onended = () => { playBtn.textContent = '▶'; fill.style.width = '0%'; };
  a.onloadedmetadata = () => { dur.textContent = fmt(a.duration); };
  a.ontimeupdate = () => {
    if (a.duration) fill.style.width = (a.currentTime / a.duration * 100) + '%';
    cur.textContent = fmt(a.currentTime);
  };
  prog.onclick = e => {
    const rect = prog.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * a.duration;
  };
  vol.oninput = () => { a.volume = vol.value / 100; };
  a.volume = 0.8;

  a.addEventListener('canplay', () => {
    a.play().then(() => { playBtn.textContent = '⏸'; }).catch(() => {});
  }, { once: true });
</script>
</body>
</html>
"""


def render_player(track_query):
    url, resolved, art = fetch_preview(track_query)
    if not url:
        st.warning(f"No preview found for: {track_query}")
        return

    mood = ""
    if st.session_state.emotion:
        mood = f"{EMOJI.get(st.session_state.emotion, '')} {st.session_state.emotion.title()}"

    art_url = art or "https://placehold.co/120x120/2a2a3e/ffffff?text=%F0%9F%8E%B5"

    rendered = (PLAYER_HTML
        .replace("__SRC__", html_lib.escape(url, quote=True))
        .replace("__ART__", html_lib.escape(art_url, quote=True))
        .replace("__TITLE__", html_lib.escape(resolved or track_query))
        .replace("__MOOD__", html_lib.escape(mood)))

    components.html(rendered, height=180)


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
            ["Emotion-recommended", "Manual playlist"],
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
                st.session_state.modality_probs = {
                    "Facial": facial_probs,
                    "Voice": audio_probs,
                    "Text": text_probs,
                }

    if st.session_state.emotion_probs:
        st.divider()
        e = st.session_state.emotion
        st.success(f"Detected emotion: **{EMOJI.get(e, '')} {e.title()}** ({st.session_state.emotion_probs[e]*100:.1f}%)")

        with st.expander("Probability breakdown (fused)"):
            for k, v in sorted(st.session_state.emotion_probs.items(), key=lambda x: -x[1]):
                st.write(f"- {EMOJI.get(k, '')} {k.title()}: {v*100:.1f}%")

        modality_probs = st.session_state.get("modality_probs", {})
        active = {k: v for k, v in modality_probs.items() if v}
        if active:
            with st.expander("Per-modality breakdown"):
                cols = st.columns(len(active))
                for col, (name, probs) in zip(cols, active.items()):
                    with col:
                        st.markdown(f"**{name}**")
                        top = max(probs, key=probs.get)
                        st.write(f"{EMOJI.get(top, '')} {top.title()}")
                        for k, v in sorted(probs.items(), key=lambda x: -x[1]):
                            st.caption(f"{k.title()}: {v*100:.1f}%")

    st.divider()
    st.subheader("2. Build your queue")

    if mode == "Emotion-recommended":
        if st.session_state.emotion:
            if st.button("🎯 Get recommendation"):
                emo = st.session_state.emotion
                song = recommender.recommend_song(emo)
                if not song:
                    st.info(f"No track in local CSV for {emo!r} — searching online instead.")
                    song = f"{emo} mood songs"
                st.session_state.queue = [song]
                st.session_state.current_idx = 0
        else:
            st.info("Detect an emotion first.")

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

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("⏮ Previous", disabled=idx == 0, use_container_width=True):
                st.session_state.current_idx -= 1
                st.rerun()
        with c2:
            if st.button("⏭ Next", disabled=idx >= total - 1, use_container_width=True, type="primary"):
                st.session_state.current_idx += 1
                st.rerun()
        with c3:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.queue = []
                st.session_state.current_idx = 0
                st.rerun()

        with st.expander(f"Queue ({total} tracks)"):
            for i, q in enumerate(st.session_state.queue):
                marker = "▶️" if i == idx else "  "
                st.write(f"{marker} {i+1}. {q}")


if __name__ == "__main__":
    main()
