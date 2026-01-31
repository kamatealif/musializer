import pygame
import librosa
import numpy as np
import sys
import time
import threading
import os

# ---------------- CONFIG ----------------
START_WIDTH, START_HEIGHT = 900, 600

SIDEBAR_WIDTH = 240  # fixed width

# 🎨 COLOR THEME
BG_COLOR = (25, 25, 28)  # #19191C
PRIMARY = (253, 53, 109)  # #FD356D

SIDEBAR_BG = (32, 32, 38)
BUTTON_BG = (45, 45, 54)
BUTTON_HOVER = (65, 65, 78)
BUTTON_ACTIVE = PRIMARY

BAR_COLOR = PRIMARY
CAP_COLOR = (255, 160, 190)

TEXT_COLOR = (235, 235, 240)
MUTED_TEXT = (160, 160, 175)

NUM_BARS = 40
FFT_SIZE = 2048
BOTTOM_MARGIN = 70

ATTACK = 0.75
DECAY = 0.90
CAP_DELAY = 0.85
CAP_GRAVITY = 0.90

# --------- AUDIO STATE ---------
playlist = []
current_track_index = -1

audio_ready = False
audio_data = None
sample_rate = None


def load_audio(index):
    global audio_ready, audio_data, sample_rate
    audio_ready = False

    path = playlist[index]
    audio_data, sample_rate = librosa.load(path, sr=22050, mono=True)

    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    audio_ready = True


def draw_sidebar(screen, font, active, mouse_pos):
    height = screen.get_height()
    pygame.draw.rect(screen, SIDEBAR_BG, (0, 0, SIDEBAR_WIDTH, height))

    title = font.render("PLAYLIST", True, MUTED_TEXT)
    screen.blit(title, (16, 16))

    y = 50
    for i, path in enumerate(playlist):
        name = os.path.basename(path)[:26]
        rect = pygame.Rect(12, y, SIDEBAR_WIDTH - 24, 36)
        hovered = rect.collidepoint(mouse_pos)

        if i == active:
            color = BUTTON_ACTIVE
        elif hovered:
            color = BUTTON_HOVER
        else:
            color = BUTTON_BG

        pygame.draw.rect(screen, color, rect, border_radius=8)
        screen.blit(
            font.render(name, True, TEXT_COLOR),
            (rect.x + 12, rect.y + 8),
        )
        y += 44


def get_clicked_track(mouse_pos):
    x, y = mouse_pos
    if x > SIDEBAR_WIDTH:
        return None
    index = (y - 50) // 44
    return index if 0 <= index < len(playlist) else None


def main():
    global current_track_index

    pygame.init()
    pygame.mixer.init(frequency=22050)

    screen = pygame.display.set_mode(
        (START_WIDTH, START_HEIGHT),
        pygame.RESIZABLE | pygame.SCALED,
    )
    pygame.display.set_caption("Music Visualizer")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Inter", 16)

    bar_heights = np.zeros(NUM_BARS)
    cap_heights = np.zeros(NUM_BARS)

    freqs = None
    log_bins = None
    start_time = time.time()

    RUNNING = True
    while RUNNING:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUNNING = False

            # 🎯 Drag & Drop MP3
            if event.type == pygame.DROPFILE:
                path = event.file
                if path.lower().endswith(".mp3"):
                    playlist.append(path)

                    if current_track_index == -1:
                        current_track_index = 0
                        threading.Thread(
                            target=load_audio,
                            args=(current_track_index,),
                            daemon=True,
                        ).start()
                        bar_heights[:] = 0
                        cap_heights[:] = 0
                        start_time = time.time()

            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked = get_clicked_track(mouse_pos)
                if clicked is not None and clicked != current_track_index:
                    current_track_index = clicked
                    threading.Thread(
                        target=load_audio,
                        args=(current_track_index,),
                        daemon=True,
                    ).start()
                    bar_heights[:] = 0
                    cap_heights[:] = 0
                    start_time = time.time()

        # Auto-play next track
        if audio_ready and not pygame.mixer.music.get_busy() and playlist:
            current_track_index = (current_track_index + 1) % len(playlist)
            threading.Thread(
                target=load_audio,
                args=(current_track_index,),
                daemon=True,
            ).start()
            bar_heights[:] = 0
            cap_heights[:] = 0
            start_time = time.time()

        WIDTH, HEIGHT = screen.get_size()
        screen.fill(BG_COLOR)

        draw_sidebar(screen, font, current_track_index, mouse_pos)

        if not playlist:
            msg = font.render("Drag & drop MP3 files here", True, MUTED_TEXT)
            screen.blit(msg, msg.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(30)
            continue

        if not audio_ready:
            loading = font.render("Loading…", True, TEXT_COLOR)
            screen.blit(loading, loading.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(30)
            continue

        if freqs is None:
            freqs = np.fft.rfftfreq(FFT_SIZE, 1 / sample_rate)
            log_bins = np.logspace(np.log10(15), np.log10(12000), NUM_BARS + 1)

        elapsed = time.time() - start_time
        idx = int(elapsed * sample_rate)

        MAX_HEIGHT = int(HEIGHT * 0.5)
        vis_width = WIDTH - SIDEBAR_WIDTH
        spacing = (vis_width * 0.85) / NUM_BARS
        bar_width = int(spacing * 0.5)
        start_x = SIDEBAR_WIDTH + (vis_width - spacing * NUM_BARS) / 2
        baseline_y = HEIGHT - BOTTOM_MARGIN

        if 0 <= idx < len(audio_data) - FFT_SIZE:
            frame = audio_data[idx : idx + FFT_SIZE]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(FFT_SIZE)))

            energies = np.zeros(NUM_BARS)
            for i in range(NUM_BARS):
                mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
                if not np.any(mask):
                    energies[i] = 0.02
                    continue

                band = spectrum[mask]
                base = np.mean(band) * 0.6 + np.max(band) * 0.4
                bass_boost = 1.8 - (i / NUM_BARS)
                energies[i] = base * bass_boost + 0.015

            energies /= np.max(energies) + 1e-6

            for i, energy in enumerate(energies):
                target = energy * MAX_HEIGHT

                if target > bar_heights[i]:
                    bar_heights[i] = bar_heights[i] * (1 - ATTACK) + target * ATTACK
                else:
                    bar_heights[i] *= DECAY

                cap_heights[i] = cap_heights[i] * CAP_DELAY + bar_heights[i] * (
                    1 - CAP_DELAY
                )

                if cap_heights[i] > bar_heights[i]:
                    cap_heights[i] *= CAP_GRAVITY

                x = start_x + i * spacing + (spacing - bar_width) / 2

                pygame.draw.rect(
                    screen,
                    BAR_COLOR,
                    (
                        int(x),
                        int(baseline_y - bar_heights[i]),
                        bar_width,
                        int(bar_heights[i]),
                    ),
                    border_radius=6,
                )

                cap_h = 6
                pygame.draw.rect(
                    screen,
                    CAP_COLOR,
                    (
                        int(x),
                        int(baseline_y - cap_heights[i] - cap_h),
                        bar_width,
                        cap_h,
                    ),
                    border_radius=3,
                )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
