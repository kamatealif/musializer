import pygame
import librosa
import numpy as np
import time
import threading
import colorsys
import warnings
import math

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
START_WIDTH, START_HEIGHT = 1280, 720
BG_COLOR = (10, 10, 12)
NUM_BARS = 50
FFT_SIZE = 2048
FPS = 60

# MOTION (smooth & expressive)
SPRING_K = 0.10
DAMPING = 0.88

# AMPLITUDE CONTROL
CEILING_MARGIN = 2
EASING_STRENGTH = 3.0
AMPLITUDE_RATIO = 0.8   # 🔥 80% vertical presence

# NEIGHBOR FOLLOW (visual continuity)
MIN_ACTIVITY = 0.015
FOLLOW_OFFSET = 3

# CURVED FLOAT MOTION
FLOAT_AMPLITUDE = 8     # pixels
FLOAT_SPEED = 1.8       # speed of curve loop

# GLOW
GLOW_LAYERS = 4
GLOW_ALPHA = 36

# LAYOUT
TOP_MARGIN = 48
BOTTOM_MARGIN = 18
GAP_BARS_TO_TIMELINE = 16
TIMELINE_HEIGHT = 8
TIME_TEXT_PADDING = 12


def format_time(seconds):
    seconds = max(0, seconds)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


class Visualizer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode(
            (START_WIDTH, START_HEIGHT),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Luma Visualizer")
        self.clock = pygame.time.Clock()

        self.audio_data = None
        self.sample_rate = None
        self.audio_duration = 1.0
        self.audio_ready = False

        self.heights = np.zeros(NUM_BARS)
        self.velocities = np.zeros(NUM_BARS)

        self.energy_history = np.zeros(40)
        self.energy_idx = 0

        self.start_time = 0.0
        self.seek_offset = 0.0

        self.time_font = pygame.font.SysFont("sans-serif", 14)

        # Phase offsets so bars don’t move together
        self.phase_offsets = np.random.uniform(0, math.pi * 2, NUM_BARS)

    # ---------------- AUDIO ----------------
    def load_audio(self, path):
        try:
            self.audio_ready = False

            self.audio_data, self.sample_rate = librosa.load(path, sr=22050)
            self.audio_duration = len(self.audio_data) / self.sample_rate

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.start_time = time.time()
            self.seek_offset = 0.0

            self.audio_ready = True
        except Exception as e:
            print("Audio load failed:", e)

    def seek(self, seconds):
        seconds = max(0.0, min(seconds, self.audio_duration - 0.1))
        pygame.mixer.music.stop()
        pygame.mixer.music.play(start=seconds)
        self.seek_offset = seconds
        self.start_time = time.time()

    # ---------------- LOOP ----------------
    def run(self):
        running = True
        while running:
            W, H = self.screen.get_size()
            self.screen.fill(BG_COLOR)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.DROPFILE:
                    threading.Thread(
                        target=self.load_audio,
                        args=(event.file,),
                        daemon=True
                    ).start()

                if event.type == pygame.MOUSEBUTTONDOWN and self.audio_ready:
                    if event.button == 1:
                        self.handle_timeline_click(event.pos, W, H)

            if not self.audio_ready:
                self.draw_ui("DROP AUDIO FILE", W, H)
            elif not pygame.mixer.music.get_busy():
                self.draw_ui("AUDIO FINISHED", W, H)
            else:
                self.render(W, H)

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    # ---------------- INPUT ----------------
    def handle_timeline_click(self, pos, W, H):
        x, y = pos
        timeline_y = H - BOTTOM_MARGIN - TIMELINE_HEIGHT
        if timeline_y <= y <= timeline_y + TIMELINE_HEIGHT:
            self.seek((x / W) * self.audio_duration)

    # ---------------- RENDER ----------------
    def render(self, W, H):
        elapsed = min(
            (time.time() - self.start_time) + self.seek_offset,
            self.audio_duration
        )

        idx = int(elapsed * self.sample_rate)
        if idx + FFT_SIZE >= len(self.audio_data):
            return

        frame = self.audio_data[idx:idx + FFT_SIZE]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(FFT_SIZE)))

        energy = np.mean(spectrum)
        self.energy_history[self.energy_idx] = energy
        self.energy_idx = (self.energy_idx + 1) % len(self.energy_history)
        adaptive_max = max(np.max(self.energy_history), 1e-6)

        timeline_y = H - BOTTOM_MARGIN - TIMELINE_HEIGHT
        bars_bottom = timeline_y - GAP_BARS_TO_TIMELINE
        bars_max_height = bars_bottom - TOP_MARGIN
        usable_height = (bars_max_height - CEILING_MARGIN) * AMPLITUDE_RATIO

        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / self.sample_rate)
        bins = np.logspace(np.log10(30), np.log10(12000), NUM_BARS + 1)

        spacing = W / NUM_BARS
        bar_w = spacing * 0.6

        targets = np.zeros(NUM_BARS)

        # ---- Audio → targets ----
        for i in range(NUM_BARS):
            mask = (freqs >= bins[i]) & (freqs < bins[i + 1])
            raw = np.mean(spectrum[mask]) if np.any(mask) else 0

            norm = raw / adaptive_max
            compressed = np.log1p(norm * 6) / np.log1p(6)
            eased = 1 - np.exp(-compressed * EASING_STRENGTH)

            targets[i] = eased * usable_height

        # ---- Neighbor follow ----
        threshold = MIN_ACTIVITY * usable_height
        for i in range(1, NUM_BARS - 1):
            if targets[i] < threshold:
                targets[i] = max(
                    (targets[i - 1] + targets[i + 1]) * 0.5 - FOLLOW_OFFSET,
                    0
                )

        t = time.time()

        # ---- Physics + draw ----
        for i in range(NUM_BARS):
            acc = (targets[i] - self.heights[i]) * SPRING_K - self.velocities[i] * DAMPING
            self.velocities[i] += acc
            self.heights[i] += self.velocities[i]

            h = self.heights[i]
            if h <= 1:
                continue

            # Curved floating motion (fast → slow → fast)
            phase = t * FLOAT_SPEED + self.phase_offsets[i]
            curve = (1 - math.cos(phase)) * 0.5
            float_offset = curve * FLOAT_AMPLITUDE * (h / usable_height)

            draw_h = int(h + float_offset)
            x = i * spacing + (spacing - bar_w) / 2
            y = bars_bottom - draw_h

            hue = i / NUM_BARS
            color = [
                int(c * 255)
                for c in colorsys.hsv_to_rgb(hue * 0.75, 0.7, 1.0)
            ]

            # Glow
            for g in range(GLOW_LAYERS, 0, -1):
                glow = pygame.Surface(
                    (bar_w + g * 8, draw_h + g * 6),
                    pygame.SRCALPHA
                )
                glow.fill((*color, GLOW_ALPHA // g))
                self.screen.blit(glow, (x - g * 4, y - g * 6))

            pygame.draw.rect(
                self.screen,
                color,
                (int(x), int(y), int(bar_w), draw_h),
                border_radius=int(bar_w // 2)
            )

        # ---- Timeline ----
        progress = elapsed / self.audio_duration
        pygame.draw.rect(
            self.screen,
            (40, 40, 45),
            (0, timeline_y, W, TIMELINE_HEIGHT),
            border_radius=TIMELINE_HEIGHT // 2
        )
        pygame.draw.rect(
            self.screen,
            (160, 220, 255),
            (0, timeline_y, int(W * progress), TIMELINE_HEIGHT),
            border_radius=TIMELINE_HEIGHT // 2
        )

        time_text = f"{format_time(elapsed)} / {format_time(self.audio_duration)}"
        txt = self.time_font.render(time_text, True, (160, 160, 170))
        self.screen.blit(
            txt,
            (W - txt.get_width() - TIME_TEXT_PADDING,
             timeline_y - txt.get_height() - 4)
        )

    def draw_ui(self, text, W, H):
        font = pygame.font.SysFont("sans-serif", 32)
        surf = font.render(text, True, (80, 80, 90))
        self.screen.blit(surf, surf.get_rect(center=(W // 2, H // 2)))


if __name__ == "__main__":
    Visualizer().run()
