import pygame
import numpy as np

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Circle")
orig = (300, 200)
BG = (0, 0, 0)
BLUE = (50, 120, 220)
vertices = [orig, orig]
running = True
angle = 0.0
variance = 30

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                variance += 1
            elif pygame.mouse.get_pressed()[2]:
                variance -= 1 if variance != 0 else 0
            print(variance)
        elif event.type == pygame.MOUSEWHEEL:
            angle += event.y
            angle = angle % 361
            print(angle)

    next_angle = np.random.normal(loc=np.deg2rad(angle), scale=np.deg2rad(variance))
    next_mag = np.random.normal(loc=3, scale=1)
    next_x = np.cos(next_angle) * next_mag
    next_y = -1 * np.sin(next_angle) * next_mag
    screen.fill(BG)
    pygame.draw.circle(screen, BLUE, vertices[-1], 6)
    pygame.draw.lines(screen, (255, 0, 0), False, vertices, 1)
    pygame.display.flip()
    vertices.append((vertices[-1][0] + next_x, vertices[-1][1] + next_y))
    clock.tick(10)

pygame.quit()
