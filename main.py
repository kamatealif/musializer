import pygame
import librosa
import numpy as np
import time
import threading
import colorsys
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
START_WIDTH, START_HEIGHT = 1280, 720
BG_COLOR = (10, 10, 12)
NUM_BARS = 50
FFT_SIZE = 2048
FPS = 60

# MOTION (RELAXED / BUTTERY)
SPRING_K = 0.10
DAMPING = 0.88

# AMPLITUDE CONTROL
CEILING_MARGIN = 2
EASING_STRENGTH = 3.0

# NEIGHBOR COUPLING
COUPLING = 0.15

# GLOW
GLOW_LAYERS = 4
GLOW_ALPHA = 38


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
        self.audio_ready = False

        self.heights = np.zeros(NUM_BARS)
        self.velocities = np.zeros(NUM_BARS)

        self.energy_history = np.zeros(40)
        self.energy_idx = 0

    # ---------------- AUDIO ----------------
    def load_audio(self, path):
        try:
            self.audio_ready = False
            self.audio_data, self.sample_rate = librosa.load(path, sr=22050)

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.start_time = time.time()

            self.audio_ready = True
        except Exception as e:
            print("Audio load failed:", e)

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

            if not self.audio_ready:
                self.draw_ui("DROP AUDIO FILE", W, H)
            elif not pygame.mixer.music.get_busy():
                self.draw_ui("AUDIO FINISHED", W, H)
            else:
                self.render_bars(W, H)

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    # ---------------- VISUALS ----------------
    def render_bars(self, W, H):
        elapsed = time.time() - self.start_time
        idx = int(elapsed * self.sample_rate)

        if idx + FFT_SIZE >= len(self.audio_data):
            return

        frame = self.audio_data[idx:idx + FFT_SIZE]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(FFT_SIZE)))

        # ---- Adaptive normalization ----
        energy = np.mean(spectrum)
        self.energy_history[self.energy_idx] = energy
        self.energy_idx = (self.energy_idx + 1) % len(self.energy_history)
        adaptive_max = max(np.max(self.energy_history), 1e-6)

        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / self.sample_rate)
        bins = np.logspace(np.log10(30), np.log10(12000), NUM_BARS + 1)

        spacing = W / NUM_BARS
        bar_w = spacing * 0.6

        BASE_Y = H * 0.8
        MAX_H = H * 0.5
        SOFT_MAX = MAX_H - CEILING_MARGIN

        # --------- STEP 1: BUILD TARGETS ---------
        targets = np.zeros(NUM_BARS)

        for i in range(NUM_BARS):
            mask = (freqs >= bins[i]) & (freqs < bins[i + 1])
            raw = np.mean(spectrum[mask]) if np.any(mask) else 0

            norm = raw / adaptive_max
            compressed = np.log1p(norm * 6) / np.log1p(6)

            eased = 1 - np.exp(-compressed * EASING_STRENGTH)
            targets[i] = eased * (1 + i / NUM_BARS * 1.2) * SOFT_MAX

        # --------- STEP 2: NEIGHBOR COUPLING ---------
        for i in range(1, NUM_BARS - 1):
            targets[i] += COUPLING * (
                targets[i - 1] + targets[i + 1] - 2 * targets[i]
            )

        # --------- STEP 3: APPLY PHYSICS + DRAW ---------
        for i in range(NUM_BARS):
            acc = (targets[i] - self.heights[i]) * SPRING_K - self.velocities[i] * DAMPING
            self.velocities[i] += acc
            self.heights[i] += self.velocities[i]

            h = int(self.heights[i])
            if h <= 1:
                continue

            x = i * spacing + (spacing - bar_w) / 2

            hue = i / NUM_BARS
            color = [
                int(c * 255)
                for c in colorsys.hsv_to_rgb(hue * 0.75, 0.7, 1.0)
            ]

            # ---- GLOW ----
            for g in range(GLOW_LAYERS, 0, -1):
                glow_h = h + g * 6
                glow = pygame.Surface(
                    (bar_w + g * 8, glow_h),
                    pygame.SRCALPHA
                )
                glow.fill((*color, GLOW_ALPHA // g))
                self.screen.blit(
                    glow,
                    (x - g * 4, BASE_Y - glow_h)
                )

            # ---- MAIN BAR ----
            pygame.draw.rect(
                self.screen,
                color,
                (int(x), int(BASE_Y - h), int(bar_w), h),
                border_radius=int(bar_w // 2)
            )

    # ---------------- UI ----------------
    def draw_ui(self, text, W, H):
        font = pygame.font.SysFont("sans-serif", 32)
        surf = font.render(text, True, (80, 80, 90))
        self.screen.blit(surf, surf.get_rect(center=(W // 2, H // 2)))


if __name__ == "__main__":
    Visualizer().run()
