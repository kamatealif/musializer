import pygame
import numpy as np
import librosa
import os
import threading
import traceback
import math

# --- Configuration ---
WIDTH, HEIGHT = 1400, 800
SIDEBAR_WIDTH = 280
FPS = 60
MAX_BARS = 256
BAR_GAP = 2
SMOOTHING_FACTOR = 0.88  # Higher = smoother (0.0-1.0)
BAR_PRESETS = [32, 64, 96, 128, 192, 256]
SCROLL_SPEED = 30  # Sidebar scroll speed

# Colors - Updated to requested scheme
COLOR_PRIMARY = (253, 53, 110)  # #FD356E - Main accent
COLOR_PRIMARY_DARK = (180, 30, 70)  # Darker shade for headers/borders
COLOR_PRIMARY_LIGHT = (255, 120, 150)  # Lighter shade for highlights
COLOR_BG = (25, 25, 28)  # #19191C - Main background
COLOR_SIDEBAR = (35, 35, 40)  # Slightly lighter than bg
COLOR_SIDEBAR_HOVER = (60, 40, 50)  # Pink-tinted hover state
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_DIM = (120, 120, 120)


class PlaylistItem:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.duration = 0  # Will be populated when loaded


class AudioVisualizer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.RESIZABLE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Spectrum Visualizer - Drop Multiple Files")
        self.clock = pygame.time.Clock()
        self.running = True
        self.show_help = False

        # Playlist system
        self.playlist = []  # List of PlaylistItem
        self.current_index = -1  # Currently playing index
        self.sidebar_scroll = 0  # Scroll offset
        self.item_height = 36

        # Audio state
        self.is_analyzing = False
        self.active_bars = 96
        self.full_spectrogram = None
        self.times = None
        self.current_duration = 0

        # Animation arrays
        self.bar_heights = np.zeros(MAX_BARS)
        self.target_heights = np.zeros(MAX_BARS)  # For smooth easing

        # Fonts
        self.font = pygame.font.SysFont("Consolas", 16)
        self.font_bold = pygame.font.SysFont("Consolas", 16, bold=True)
        self.big_font = pygame.font.SysFont("Consolas", 32, bold=True)
        self.time_font = pygame.font.SysFont("Consolas", 22, bold=True)
        self.sidebar_font = pygame.font.SysFont("Consolas", 14)
        self.help_font = pygame.font.SysFont("Consolas", 22)
        self.title_font = pygame.font.SysFont("Consolas", 28, bold=True)

    def format_time(self, seconds):
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def add_to_playlist(self, path):
        """Add file to playlist if valid"""
        ext = os.path.splitext(path)[1].lower()
        if ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"]:
            item = PlaylistItem(path)
            self.playlist.append(item)
            # If first item, auto-play
            if len(self.playlist) == 1:
                self.play_index(0)
            return True
        return False

    def play_index(self, index):
        """Play specific playlist index"""
        if 0 <= index < len(self.playlist):
            self.current_index = index
            self.load_file_threaded(self.playlist[index].path)

    def load_file_threaded(self, path):
        if self.is_analyzing:
            return
        self.is_analyzing = True
        thread = threading.Thread(target=self._analyze_audio, args=(path,))
        thread.daemon = True
        thread.start()

    def _analyze_audio(self, path):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(path)

            # Quick metadata load for duration
            y, sr = librosa.load(path, sr=None, mono=True)
            self.current_duration = librosa.get_duration(y=y, sr=sr)

            # Update playlist duration
            if 0 <= self.current_index < len(self.playlist):
                self.playlist[self.current_index].duration = self.current_duration

            # Full analysis for visualization
            hop_length = 512
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=MAX_BARS, hop_length=hop_length
            )
            self.full_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)
            self.times = librosa.frames_to_time(
                np.arange(self.full_spectrogram.shape[1]), sr=sr, hop_length=hop_length
            )

            # Reset animation
            self.bar_heights.fill(0)
            self.target_heights.fill(0)

            pygame.mixer.music.play()
            # Set end event to auto-advance
            pygame.mixer.music.set_endevent(pygame.USEREVENT + 1)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            print(traceback.format_exc())
        finally:
            self.is_analyzing = False

    def get_current_bars(self):
        if self.full_spectrogram is None:
            return None

        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:
            if not pygame.mixer.music.get_busy() and self.current_duration > 0:
                pass
            return np.zeros(self.active_bars)

        idx = np.searchsorted(self.times, pos_ms / 1000.0)
        idx = min(idx, self.full_spectrogram.shape[1] - 1)

        full_frame = self.full_spectrogram[:, idx]
        frame = np.clip((full_frame - (-80)) / 80, 0, 1)
        frame = np.power(frame, 0.6)

        if self.active_bars == MAX_BARS:
            return frame
        else:
            bin_size = MAX_BARS // self.active_bars
            return (
                frame[: self.active_bars * bin_size]
                .reshape(self.active_bars, bin_size)
                .mean(axis=1)
            )

    def update_animation(self):
        """Buttery smooth easing"""
        if self.full_spectrogram is None:
            return

        data = self.get_current_bars()
        if data is None:
            return

        # Moved down to match new visual position (was 0.75)
        available_height = int(self.screen.get_height() * 0.85) - 100

        for i in range(self.active_bars):
            target = data[i] * available_height
            self.target_heights[i] = target

            diff = target - self.bar_heights[i]
            self.bar_heights[i] += diff * (1 - SMOOTHING_FACTOR)

            if abs(diff) < 0.5 and target < 1:
                self.bar_heights[i] = 0

    def truncate_to_two_words(self, text):
        """Show only first 2 words, add ... if there are more"""
        words = text.replace("_", " ").replace("-", " ").split()
        if len(words) > 2:
            return " ".join(words[:2]) + "..."
        return " ".join(words)

    def draw_sidebar(self):
        """Draw playlist sidebar with 2-word truncation"""
        h = self.screen.get_height()

        # Background
        pygame.draw.rect(self.screen, COLOR_SIDEBAR, (0, 0, SIDEBAR_WIDTH, h))
        pygame.draw.line(
            self.screen,
            COLOR_PRIMARY_DARK,
            (SIDEBAR_WIDTH - 2, 0),
            (SIDEBAR_WIDTH - 2, h),
            2,
        )

        # Header
        pygame.draw.rect(self.screen, COLOR_PRIMARY_DARK, (0, 0, SIDEBAR_WIDTH, 50))
        title = self.font_bold.render("PLAYLIST", True, COLOR_PRIMARY_LIGHT)
        self.screen.blit(title, (15, 15))

        count_text = self.font.render(
            f"{len(self.playlist)} tracks", True, COLOR_TEXT_DIM
        )
        self.screen.blit(count_text, (SIDEBAR_WIDTH - count_text.get_width() - 15, 18))

        if not self.playlist:
            empty = self.font.render("Drop files here", True, COLOR_TEXT_DIM)
            self.screen.blit(empty, (15, 70))
            return

        # Playlist items
        y_start = 60 - self.sidebar_scroll
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for idx, item in enumerate(self.playlist):
            y = y_start + idx * self.item_height

            # Skip if off-screen
            if y < 50 or y > h - 50:
                continue

            # Hover/selection check
            is_hovered = (
                0 <= mouse_x < SIDEBAR_WIDTH and y <= mouse_y < y + self.item_height
            )
            is_playing = idx == self.current_index

            if is_playing:
                pygame.draw.rect(
                    self.screen,
                    COLOR_PRIMARY,
                    (5, y, SIDEBAR_WIDTH - 10, self.item_height - 2),
                    border_radius=4,
                )
                text_color = (255, 255, 255)
                name_font = self.font_bold
            elif is_hovered:
                pygame.draw.rect(
                    self.screen,
                    COLOR_SIDEBAR_HOVER,
                    (5, y, SIDEBAR_WIDTH - 10, self.item_height - 2),
                    border_radius=4,
                )
                text_color = COLOR_PRIMARY_LIGHT
                name_font = self.font
            else:
                text_color = COLOR_TEXT
                name_font = self.font

            # Track number
            num_text = self.font.render(f"{idx + 1:2d}", True, COLOR_TEXT_DIM)
            self.screen.blit(num_text, (12, y + 8))

            # Track name - truncate to 2 words max with padding check
            max_text_width = (
                SIDEBAR_WIDTH - 100
            )  # Reserve space for number and duration
            name = self.truncate_to_two_words(item.name)
            name_surf = name_font.render(name, True, text_color)

            # Extra safety: if still too wide, truncate with ellipsis
            if name_surf.get_width() > max_text_width:
                while name_surf.get_width() > max_text_width and len(name) > 3:
                    name = name[:-4] + "..."
                    name_surf = name_font.render(name, True, text_color)

            self.screen.blit(name_surf, (40, y + 8))

            # Duration if known
            if item.duration > 0:
                dur_text = self.sidebar_font.render(
                    self.format_time(item.duration), True, COLOR_TEXT_DIM
                )
                self.screen.blit(
                    dur_text, (SIDEBAR_WIDTH - dur_text.get_width() - 12, y + 10)
                )

        # Scroll indicator
        total_height = len(self.playlist) * self.item_height
        visible_height = h - 60
        if total_height > visible_height:
            scroll_h = visible_height * (visible_height / total_height)
            scroll_pos = 60 + (
                self.sidebar_scroll / (total_height - visible_height)
            ) * (visible_height - scroll_h)
            pygame.draw.rect(
                self.screen,
                COLOR_PRIMARY,
                (SIDEBAR_WIDTH - 6, scroll_pos, 4, max(20, scroll_h)),
                border_radius=2,
            )

    def draw_spectrum(self):
        """Draw the spectrum analyzer"""
        w, h = self.screen.get_size()
        spectrum_w = w - SIDEBAR_WIDTH - 40
        start_x = SIDEBAR_WIDTH + 20

        # MOVED DOWN: Position at 85% of screen height instead of previous calculation
        bottom_y = int(h * 0.85)

        if self.full_spectrogram is None or self.active_bars == 0:
            return

        bar_total_w = spectrum_w / self.active_bars
        bar_w = max(2, bar_total_w - BAR_GAP)

        color_step = 300 / self.active_bars

        for i in range(self.active_bars):
            x = start_x + i * bar_total_w
            bar_h = self.bar_heights[i]

            if bar_h < 0.5:
                continue

            hue = i * color_step
            color = pygame.Color(0)
            color.hsva = (hue, 95, 100, 100)

            # Glow effect
            if bar_h > 5:
                glow_w = bar_w + 6
                glow_surf = pygame.Surface((int(glow_w), int(bar_h)), pygame.SRCALPHA)
                glow_color = (*color[:3], 60)
                glow_surf.fill(glow_color)
                self.screen.blit(glow_surf, (x - 3, bottom_y - bar_h))

            # Main bar
            pygame.draw.rect(
                self.screen, color, (x, bottom_y - bar_h, bar_w, bar_h), border_radius=2
            )

            # Top highlight
            if bar_h > 3:
                highlight_h = min(3, bar_h)
                highlight_color = pygame.Color(0)
                highlight_color.hsva = (hue, 30, 100, 100)
                pygame.draw.rect(
                    self.screen,
                    highlight_color,
                    (x, bottom_y - bar_h, bar_w, highlight_h),
                    border_radius=1,
                )

    def draw_progress_bar(self):
        """Draw thinner progress bar - positioned closer to spectrum"""
        w, h = self.screen.get_size()

        # Thinner progress bar
        bar_h = 6
        margin = 100
        bar_x = SIDEBAR_WIDTH + margin
        bar_w = w - SIDEBAR_WIDTH - (margin * 2)

        # MOVED DOWN: Match spectrum position at 85% + small gap
        spectrum_bottom = int(h * 0.85)
        bar_y = spectrum_bottom + 20  # 20px gap below spectrum

        # Background
        pygame.draw.rect(
            self.screen, (40, 40, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=3
        )

        if self.current_duration > 0:
            pos_s = max(0, pygame.mixer.music.get_pos() / 1000.0)
            progress = min(1.0, pos_s / self.current_duration)
            fill_w = int(bar_w * progress)

            is_playing = pygame.mixer.music.get_busy()
            fill_color = COLOR_PRIMARY if is_playing else (100, 60, 60)

            if fill_w > 0:
                pygame.draw.rect(
                    self.screen,
                    fill_color,
                    (bar_x, bar_y, fill_w, bar_h),
                    border_radius=3,
                )

            # Time labels
            curr_surf = self.time_font.render(
                self.format_time(pos_s), True, COLOR_PRIMARY_LIGHT
            )
            self.screen.blit(curr_surf, (bar_x - 90, bar_y - 9))

            rem = self.current_duration - pos_s
            rem_surf = self.time_font.render(
                f"-{self.format_time(rem)}", True, COLOR_PRIMARY_LIGHT
            )
            self.screen.blit(rem_surf, (bar_x + bar_w + 15, bar_y - 9))

    def draw_help_overlay(self):
        w, h = self.screen.get_size()

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))
        self.screen.blit(overlay, (0, 0))

        box_w, box_h = 700, 500
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2

        pygame.draw.rect(
            self.screen, COLOR_SIDEBAR, (box_x, box_y, box_w, box_h), border_radius=16
        )
        pygame.draw.rect(
            self.screen,
            COLOR_PRIMARY,
            (box_x, box_y, box_w, box_h),
            width=3,
            border_radius=16,
        )

        title = self.title_font.render("CONTROLS", True, COLOR_PRIMARY_LIGHT)
        self.screen.blit(title, title.get_rect(center=(w // 2, box_y + 50)))

        controls = [
            ("UP / DOWN", "Change spectrum bars (32 → 256)"),
            ("LEFT / RIGHT", "Previous / Next track"),
            ("SPACE", "Play / Pause"),
            ("CLICK", "Select track from sidebar"),
            ("SCROLL", "Scroll playlist"),
            ("H", "Toggle this help"),
            ("ESC", "Exit"),
            ("", ""),
            ("MOUSE", ""),
            ("Drag & Drop", "Add files to playlist"),
        ]

        y = box_y + 110
        for key, desc in controls:
            if key == "":
                y += 20
                continue

            key_surf = self.help_font.render(key, True, COLOR_PRIMARY_LIGHT)
            self.screen.blit(key_surf, (box_x + 60, y))

            desc_surf = self.help_font.render(desc, True, COLOR_TEXT)
            self.screen.blit(desc_surf, (box_x + 280, y))

            y += 45

        hint = self.font.render("Press H or ESC to close", True, COLOR_TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(w // 2, box_y + box_h - 40)))

    def handle_click(self, pos):
        """Handle mouse clicks on sidebar"""
        x, y = pos

        if x >= SIDEBAR_WIDTH:
            return

        if y < 60:
            return

        rel_y = y - 60 + self.sidebar_scroll
        idx = rel_y // self.item_height

        if 0 <= idx < len(self.playlist):
            self.play_index(idx)

    def draw(self):
        if not self.is_analyzing:
            self.update_animation()

        self.screen.fill(COLOR_BG)

        self.draw_sidebar()
        self.draw_spectrum()
        self.draw_progress_bar()

        if self.is_analyzing:
            w, h = self.screen.get_size()
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text = self.big_font.render("LOADING...", True, COLOR_PRIMARY)
            self.screen.blit(text, text.get_rect(center=(w // 2, h // 2)))

        if self.show_help:
            self.draw_help_overlay()

        pygame.display.flip()

    def run(self):
        print("Spectrum Visualizer with Playlist")
        print("Drop multiple files to create playlist")
        print("Controls: UP/DOWN (bars), LEFT/RIGHT (tracks), SPACE (pause), H (help)")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.USEREVENT + 1:
                    if self.current_index < len(self.playlist) - 1:
                        self.play_index(self.current_index + 1)

                elif event.type == pygame.DROPFILE:
                    path = event.file
                    self.add_to_playlist(path)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_help:
                            self.show_help = False
                        else:
                            self.running = False

                    elif event.key == pygame.K_SPACE:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()

                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help

                    elif event.key == pygame.K_LEFT:
                        if self.current_index > 0:
                            self.play_index(self.current_index - 1)

                    elif event.key == pygame.K_RIGHT:
                        if self.current_index < len(self.playlist) - 1:
                            self.play_index(self.current_index + 1)

                    elif event.key == pygame.K_UP:
                        idx = (
                            BAR_PRESETS.index(self.active_bars)
                            if self.active_bars in BAR_PRESETS
                            else 0
                        )
                        if idx < len(BAR_PRESETS) - 1:
                            self.active_bars = BAR_PRESETS[idx + 1]
                            self.bar_heights.fill(0)
                            self.target_heights.fill(0)

                    elif event.key == pygame.K_DOWN:
                        idx = (
                            BAR_PRESETS.index(self.active_bars)
                            if self.active_bars in BAR_PRESETS
                            else len(BAR_PRESETS) - 1
                        )
                        if idx > 0:
                            self.active_bars = BAR_PRESETS[idx - 1]
                            self.bar_heights.fill(0)
                            self.target_heights.fill(0)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                    elif event.button == 4:
                        self.sidebar_scroll = max(0, self.sidebar_scroll - SCROLL_SPEED)
                    elif event.button == 5:
                        max_scroll = max(
                            0,
                            len(self.playlist) * self.item_height
                            - self.screen.get_height()
                            + 100,
                        )
                        self.sidebar_scroll = min(
                            max_scroll, self.sidebar_scroll + SCROLL_SPEED
                        )

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    app = AudioVisualizer()
    app.run()
