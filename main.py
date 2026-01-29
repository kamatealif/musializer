import pygame

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 900, 600
SIDEBAR_WIDTH = 200

BG_COLOR = (15, 8, 12)
SIDEBAR_COLOR = (25, 12, 18)
BTN_COLOR = (50, 25, 35)
BTN_HOVER = (80, 35, 55)
TEXT_COLOR = (240, 240, 240)
ACCENT = (180, 80, 120)

MUSIC_DIR = "assets/musics"
SUPPORTED = (".wav", ".mp3", ".ogg")
RUNNING = True

if __name__ == "__main__":
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    while RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUNNING = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    RUNNING = False

        screen.fill(BG_COLOR)

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
