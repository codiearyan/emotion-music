"""
Main Multimodal Emotion-Based Music Player
Integrates all components
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.emotion_detection.facial_emotion import FacialEmotionDetector
from src.emotion_detection.audio_emotion import AudioEmotionDetector
from src.emotion_detection.text_emotion import TextEmotionDetector
from src.fusion.multimodal_fusion import MultimodalFusion
from src.music_analysis.music_emotion_recognition import MusicEmotionAnalyzer
from src.recommendation.recommendation_engine import MusicRecommendationEngine
import pygame
import time
import tempfile
import urllib.parse
import urllib.request
import json
import shutil
import subprocess


class MultimodalMusicPlayer:
    def __init__(self):
        print("🎵 Initializing Multimodal Music Player...")
        
        # Initialize all detectors
        self.facial_detector = FacialEmotionDetector()
        self.audio_detector = AudioEmotionDetector()
        self.text_detector = TextEmotionDetector()
        
        # Initialize fusion
        self.fusion = MultimodalFusion(fusion_method='attention')
        
        # Initialize music analyzer
        self.music_analyzer = MusicEmotionAnalyzer()
        
        # Initialize recommendation engine
        self.recommender = MusicRecommendationEngine(self.music_analyzer)
        
        # Initialize pygame for music
        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"⚠️  Audio device not found: {e}")
            print("   Switching to dummy audio driver for simulation.")
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            pygame.mixer.init()
        pygame.font.init()
        pygame.display.init()
        
        print("✅ All systems ready!\n")
        
    def detect_emotions(self, use_facial=True, use_audio=True, use_text=True):
        """Detect emotions from all available modalities"""
        facial_probs = None
        audio_probs = None
        text_probs = None
        
        print("\n" + "="*60)
        print("🎭 MULTIMODAL EMOTION DETECTION")
        print("="*60)
        
        # Facial emotion detection
        if use_facial:
            print("\n📹 Starting facial emotion detection...")
            print("   Look at the camera. Press ESC to finish.")
            emotion_code = self.facial_detector.detect_from_webcam(num_predictions=10)
            
            # Convert to probabilities
            facial_probs = {
                'angry': 0.9 if emotion_code == '1' else 0.05,
                'happy': 0.9 if emotion_code == '2' else 0.05,
                'neutral': 0.8 if emotion_code == '3' else 0.1,
                'sad': 0.1
            }
            print(f"   ✓ Facial: {max(facial_probs, key=facial_probs.get)}")
            
        # Audio emotion detection
        if use_audio:
            try:
                print("\n🎤 Starting audio emotion detection...")
                emotion, audio_probs = self.audio_detector.detect_from_microphone()
                print(f"   ✓ Audio: {emotion}")
            except Exception as e:
                print(f"   ⚠️  Audio detection skipped: {e}")
                audio_probs = None
                
        # Text emotion detection
        if use_text:
            print("\n💬 Text emotion analysis:")
            print("   (Press Enter to skip)")
            text_input = input("   Your text: ").strip()
            if text_input:
                emotion, text_probs = self.text_detector.detect_from_input(text_input)
                print(f"   ✓ Text: {emotion}")
            else:
                print("   ⊗ Text detection skipped")
                text_probs = None
                
        return facial_probs, audio_probs, text_probs
        
    def fuse_and_recommend(self, facial_probs, audio_probs, text_probs):
        """Fuse emotions and recommend music"""
        # Fuse emotions
        fused_probs = self.fusion.fuse_emotions(facial_probs, audio_probs, text_probs)
        
        # Display results
        detected_emotion = self.fusion.print_fusion_results(fused_probs)
        
        # Get recommendation
        print("\n" + "="*60)
        print("🎯 MUSIC RECOMMENDATION")
        print("="*60)
        
        explanation = self.recommender.get_recommendation_explanation(detected_emotion)
        print(f"\n{explanation}")
        
        song = self.recommender.recommend_song(detected_emotion)
        
        return detected_emotion, song
        
    def fetch_preview_url(self, query):
        q = urllib.parse.quote(query)
        try:
            with urllib.request.urlopen(f"https://itunes.apple.com/search?term={q}&media=music&limit=1", timeout=8) as r:
                data = json.loads(r.read())
            if data.get("results"):
                track = data["results"][0]
                if track.get("previewUrl"):
                    return track["previewUrl"], f'{track["artistName"]} - {track["trackName"]}'
        except Exception as e:
            print(f"   iTunes lookup failed: {e}")

        try:
            with urllib.request.urlopen(f"https://api.deezer.com/search?q={q}&limit=1", timeout=8) as r:
                data = json.loads(r.read())
            if data.get("data"):
                track = data["data"][0]
                if track.get("preview"):
                    return track["preview"], f'{track["artist"]["name"]} - {track["title"]}'
        except Exception as e:
            print(f"   Deezer lookup failed: {e}")

        return None, None

    def download_preview(self, url):
        path = urllib.parse.urlparse(url).path.lower()
        is_aac = path.endswith(".m4a") or path.endswith(".aac")
        suffix = ".m4a" if is_aac else ".mp3"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                tmp.write(r.read())
            tmp.close()
        except Exception as e:
            tmp.close()
            os.unlink(tmp.name)
            raise e

        if not is_aac:
            return tmp.name

        if not shutil.which("ffmpeg"):
            print("   ⚠️  ffmpeg not found — install with `brew install ffmpeg` for iTunes previews.")
            os.unlink(tmp.name)
            return None

        mp3_path = tmp.name.replace(".m4a", ".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                 "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  ffmpeg conversion failed: {e}")
            os.unlink(tmp.name)
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        return mp3_path

    EMOJI_MAP = {'angry': '😠', 'happy': '😊', 'neutral': '😐', 'sad': '😢'}

    def _play_one(self, query, detected_emotion, screen, font, emoji_font, position_text=""):
        print(f"\n🔎 Searching online for: {query}")
        preview_url, resolved_name = self.fetch_preview_url(query)
        if not preview_url:
            print(f"⚠️  No preview found for: {query}")
            return "skip"

        print(f"   ✓ Found: {resolved_name}")

        try:
            song_path = self.download_preview(preview_url)
        except Exception as e:
            print(f"⚠️  Failed to download preview: {e}")
            return "skip"
        if not song_path:
            return "skip"

        display_name = resolved_name or query
        action = "finished"

        try:
            pygame.mixer.music.load(song_path)
            print(f"\n🎵 Now Playing: {display_name}")
            pygame.mixer.music.play()
            pygame.time.wait(300)

            clock = pygame.time.Clock()
            emoji_char = self.EMOJI_MAP.get(detected_emotion, '😐')
            paused = False

            while pygame.mixer.music.get_busy() or paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        action = "quit"
                        pygame.mixer.music.stop()
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            pygame.mixer.music.pause()
                            paused = True
                            print("⏸️  Paused")
                        elif event.key == pygame.K_r:
                            pygame.mixer.music.unpause()
                            paused = False
                            print("▶️  Resumed")
                        elif event.key == pygame.K_n:
                            action = "next"
                            pygame.mixer.music.stop()
                            print("⏭️  Next")
                            break
                        elif event.key in (pygame.K_q, pygame.K_s):
                            action = "quit"
                            pygame.mixer.music.stop()
                            break

                if action != "finished":
                    break

                screen.fill((30, 30, 40))
                try:
                    emoji_surface = emoji_font.render(emoji_char, True, (255, 255, 255))
                    screen.blit(emoji_surface, emoji_surface.get_rect(center=(300, 130)))
                except Exception:
                    fallback = font.render(f"Emotion: {detected_emotion.title()}", True, (255, 200, 100))
                    screen.blit(fallback, (50, 130))

                song_text = font.render(f"{display_name[:48]}", True, (255, 255, 255))
                screen.blit(song_text, song_text.get_rect(center=(300, 240)))

                if position_text:
                    pos = font.render(position_text, True, (180, 180, 180))
                    screen.blit(pos, pos.get_rect(center=(300, 280)))

                status = "PAUSED" if paused else "PLAYING"
                status_color = (255, 200, 100) if paused else (100, 255, 100)
                status_text = font.render(status, True, status_color)
                screen.blit(status_text, status_text.get_rect(center=(300, 320)))

                hint = font.render("P:Pause  R:Resume  N:Next  Q:Quit", True, (150, 150, 150))
                hint = pygame.transform.scale(hint, (int(hint.get_width() * 0.7), int(hint.get_height() * 0.7)))
                screen.blit(hint, hint.get_rect(center=(300, 370)))

                pygame.display.flip()
                clock.tick(30)

        except Exception as e:
            print(f"⚠️  Error playing music: {e}")
        finally:
            try:
                os.unlink(song_path)
            except Exception:
                pass

        return action

    def play_queue(self, queries, detected_emotion):
        if not queries:
            print("⚠️  Queue is empty")
            return

        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption(f"SyncIn Player - {detected_emotion.title()}")
        font = pygame.font.Font(None, 32)
        emoji_font = pygame.font.SysFont("segoeuiemoji", 100)

        print("\n" + "="*60)
        print(f"Playing queue ({len(queries)} tracks)")
        print("Controls: P=Pause  R=Resume  N=Next  Q=Quit")
        print("="*60)

        for idx, query in enumerate(queries, start=1):
            position = f"Track {idx} / {len(queries)}"
            action = self._play_one(query, detected_emotion, screen, font, emoji_font, position)
            if action == "quit":
                break

        pygame.mixer.music.stop()
        pygame.quit()
        print("\n✅ Playback finished")
            
    def choose_mode(self):
        print("\n" + "="*60)
        print("🎵 MULTIMODAL EMOTION-BASED MUSIC PLAYER")
        print("="*60)
        print("\nWhat would you like to play?")
        print("  1) Emotion-recommended song (default)")
        print("  2) Search a specific song")
        print("  3) Manual playlist (paste song names)")
        choice = input("\nChoose [1/2/3] (default 1): ").strip() or "1"

        if choice == "2":
            query = input("Song name: ").strip()
            return ("search", [query]) if query else ("emotion", None)

        if choice == "3":
            print("Enter song names, one per line. Blank line to finish.")
            print("Tip: 'Artist - Title' gives best matches.")
            queue = []
            while True:
                line = input(f"  {len(queue)+1}> ").strip()
                if not line:
                    break
                queue.append(line)
            if not queue:
                return ("emotion", None)
            print(f"   ✓ Queue has {len(queue)} tracks")
            return ("manual", queue)

        return ("emotion", None)

    def run(self):
        """Main application loop"""
        mode, queue = self.choose_mode()

        input("\nPress Enter to start emotion detection...")

        facial_probs, audio_probs, text_probs = self.detect_emotions(
            use_facial=True, use_audio=True, use_text=True
        )

        detected_emotion, recommended_song = self.fuse_and_recommend(
            facial_probs, audio_probs, text_probs
        )

        if mode == "emotion":
            queue = [recommended_song] if recommended_song else []

        if not queue:
            print("⚠️  Nothing to play")
        else:
            self.play_queue(queue, detected_emotion)

        print("\n👋 Thank you for using the Multimodal Music Player!")

def main():
    try:
        player = MultimodalMusicPlayer()
        player.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
