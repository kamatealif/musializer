
import pygame
import numpy as np
import librosa
import os
import threading
import cv2
from datetime import timedelta

# ───────────────── CONFIG ─────────────────
FPS = 60
MAX_BARS = 128
BAR_GAP = 3

SPRING = 0.2     # responsiveness
DAMPING = 0.8    # smoothness

COLOR_BG = (10, 10, 14)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_DIM = (120, 120, 120)
COLOR_ACCENT = (255, 60, 120)

# ───────────────── VIDEO RENDERER ─────────────────
class VideoRenderer:
    def __init__(self, w, h, fps):
        self.w = w
        self.h = h
        self.fps = fps
        self.frame = 0
        self.total = 0
        self.writer = None

    def start(self, path, duration):
        self.total = int(duration * self.fps)
        self.writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.w, self.h)
        )
        self.frame = 0
        print(f"[RENDER] START → {path}")

    def write(self, surface):
        frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[:2] != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h))
        self.writer.write(frame)
        self.frame += 1

    def stop(self):
        self.writer.release()
        print("[RENDER] COMPLETE ✔")

# ───────────────── VISUALIZER ─────────────────
class AudioVisualizer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 512)

        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Minimal Audio Visualizer")

        self.clock = pygame.time.Clock()
        self.running = True

        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.font_big = pygame.font.SysFont("Segoe UI", 36, bold=True)

        self.file = None
        self.spec = None
        self.times = None
        self.duration = 0

        self.active_bars = 64
        self.heights = np.zeros(MAX_BARS)
        self.velocity = np.zeros(MAX_BARS)

        self.start_ticks = 0
        self.pause_time = 0.0
        self.paused = False
        self.current_time = 0.0

        self.renderer = None
        self.rendering = False

    # ───────── AUDIO LOAD ─────────
    def analyze(self, path):
        print(f"[LOAD] {os.path.basename(path)}")

        y, sr = librosa.load(path, sr=None, mono=True)
        self.duration = librosa.get_duration(y=y, sr=sr)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=MAX_BARS,
            hop_length=512
        )

        self.spec = librosa.power_to_db(mel, ref=np.max)
        self.times = librosa.frames_to_time(
            np.arange(self.spec.shape[1]),
            sr=sr,
            hop_length=512
        )

        self.file = path
        self.heights.fill(0)
        self.velocity.fill(0)

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        self.start_ticks = pygame.time.get_ticks()
        self.pause_time = 0.0
        self.paused = False

        print("[READY] SPACE=Pause | R=Render | ESC=Exit")

    def spectrum_at(self, t):
        idx = np.searchsorted(self.times, t)
        idx = min(idx, len(self.times) - 1)
        frame = self.spec[:, idx]
        frame = np.clip((frame + 80) / 80, 0, 1) ** 0.65
        return frame[:self.active_bars]

    # ───────── UPDATE ─────────
    def update(self):
        if self.spec is None:
            return

        if self.rendering:
            t = self.renderer.frame / FPS
            if t >= self.duration:
                self.renderer.stop()
                self.rendering = False
                return
        else:
            t = self.pause_time if self.paused else (
                (pygame.time.get_ticks() - self.start_ticks) / 1000
            )

        self.current_time = min(max(t, 0.0), self.duration)

        data = self.spectrum_at(self.current_time)
        max_h = self.screen.get_height() * 0.65

        for i in range(self.active_bars):
            target = data[i] * max_h
            force = (target - self.heights[i]) * SPRING
            self.velocity[i] = (self.velocity[i] + force) * DAMPING
            self.heights[i] += self.velocity[i]

    # ───────── DRAW ─────────
    def draw(self):
        self.screen.fill(COLOR_BG)
        w, h = self.screen.get_size()

        if self.spec is None:
            txt = self.font_big.render("DROP AUDIO FILE", True, COLOR_TEXT)
            self.screen.blit(txt, (w//2 - txt.get_width()//2, h//2))
            pygame.display.flip()
            return

        bar_area_w = w - 120
        bar_w = bar_area_w / self.active_bars
        base_y = h * 0.75

        for i in range(self.active_bars):
            bh = self.heights[i]
            if bh < 1:
                continue

            hue = int((i / self.active_bars) * 360)
            color = pygame.Color(0)
            color.hsva = (hue, 80, 100, 100)

            pygame.draw.rect(
                self.screen,
                color,
                (
                    60 + i * bar_w,
                    base_y - bh,
                    bar_w - BAR_GAP,
                    bh
                )
            )

        # ───── Progress Bar ─────
        bar_y = h - 40
        bar_x = 120
        bar_w2 = w - 240
        bar_h = 5

        pygame.draw.rect(
            self.screen, (40, 40, 40),
            (bar_x, bar_y, bar_w2, bar_h),
            border_radius=3
        )

        progress = self.current_time / self.duration if self.duration > 0 else 0
        fill = int(bar_w2 * progress)

        pygame.draw.rect(
            self.screen, COLOR_ACCENT,
            (bar_x, bar_y, fill, bar_h),
            border_radius=3
        )

        time_txt = f"{timedelta(seconds=int(self.current_time))} / {timedelta(seconds=int(self.duration))}"
        ts = self.font.render(time_txt, True, COLOR_TEXT_DIM)
        self.screen.blit(ts, (w//2 - ts.get_width()//2, bar_y - 18))

        pygame.display.flip()

        if self.rendering:
            self.renderer.write(self.screen)

    # ───────── RUN ─────────
    def run(self):
        print("MINIMAL VISUALIZER READY")

        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False

                elif e.type == pygame.DROPFILE and not self.rendering:
                    threading.Thread(target=self.analyze, args=(e.file,), daemon=True).start()

                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        self.running = False

                    elif e.key == pygame.K_SPACE and self.spec is not None:
                        if not self.paused:
                            pygame.mixer.music.pause()
                            self.pause_time = self.current_time
                            self.paused = True
                        else:
                            pygame.mixer.music.unpause()
                            self.start_ticks = pygame.time.get_ticks() - int(self.pause_time * 1000)
                            self.paused = False

                    elif e.key == pygame.K_r and self.spec is not None and not self.rendering:
                        out = os.path.splitext(self.file)[0] + "_viz.mp4"
                        self.renderer = VideoRenderer(1920, 1080, FPS)
                        self.renderer.start(out, self.duration)
                        self.rendering = True

            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        print("[EXIT]")


if __name__ == "__main__":
    AudioVisualizer().run()

