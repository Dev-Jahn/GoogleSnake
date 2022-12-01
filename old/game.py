import pygame
import argparse
import random


def reset_game():
    global map, snake_head_width, snake_head_height, snake_tail_width, snake_tail_height, apple_width, apple_height, snake_length
    map = [[[0, 0] for __ in range(w)] for _ in range(h)]
    snake_head_width, snake_head_height = 2, h // 2 - 1
    snake_tail_width, snake_tail_height = 1, h // 2 - 1
    apple_width, apple_height = w - 2, h // 2 - 1
    map[snake_head_height][snake_head_width] = [2, -4]  # head
    map[snake_tail_height][snake_tail_width] = [2, 2]  # tail
    map[apple_height][apple_width] = [1, 0]  # food
    snake_length = 2


def draw_screen():
    GREEN1, GREEN2 = (170, 215, 81), (162, 209, 73)
    screen.fill(GREEN1)
    for col in range(w):
        for row in range(col % 2, h, 2):
            pygame.draw.rect(screen, GREEN2, (col * tile_width, row * tile_height, tile_width, tile_height))


def generate_apple():
    # draw on screen
    screen.blit(appleImg, (apple_width * tile_width, apple_height * tile_height))


def draw_snake():
    BLUE = (71, 117, 235)
    BLACK = (0, 0, 0)
    draw_width = snake_tail_width
    draw_height = snake_tail_height
    while True:
        dir = map[draw_height][draw_width][1]
        if dir == 1:    pygame.draw.rect(screen, BLUE, (
            draw_width * tile_width + connect_width // 2, draw_height * tile_height - connect_height // 2, snake_width,
            snake_height + connect_height))
        if dir == 2:    pygame.draw.rect(screen, BLUE, (
            draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2,
            snake_width + connect_width, snake_height))
        if dir == 3:    pygame.draw.rect(screen, BLUE, (
            draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2, snake_width,
            snake_height + connect_height))
        if dir == 4:    pygame.draw.rect(screen, BLUE, (
            draw_width * tile_width - connect_width // 2, draw_height * tile_height + connect_height // 2,
            snake_width + connect_width, snake_height))
        draw_width = draw_width + dir_width[dir]
        draw_height = draw_height + dir_height[dir]
        if map[draw_height][draw_width][1] < 0:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2,
                snake_width,
                snake_height))

            # snake eyes
            eh, ew = 9, 6
            head_dir = -map[draw_height][draw_width][1]
            if head_dir % 2 == 0:
                pygame.draw.rect(screen, BLACK, (draw_width * tile_width + tile_width // 2 + eh * (head_dir // 2 - 2),
                                                 draw_height * tile_height + tile_height // 2 - (ew * 3 // 2), eh, ew))
                pygame.draw.rect(screen, BLACK, (draw_width * tile_width + tile_width // 2 + eh * (head_dir // 2 - 2),
                                                 draw_height * tile_height + tile_height // 2 + (ew // 2), eh, ew))
            else:
                pygame.draw.rect(screen, BLACK, (draw_width * tile_width + tile_width // 2 - (ew * 3 // 2),
                                                 draw_height * tile_height + tile_height // 2 - eh * (head_dir // 2),
                                                 ew, eh))
                pygame.draw.rect(screen, BLACK, (draw_width * tile_width + tile_width // 2 + (ew // 2),
                                                 draw_height * tile_height + tile_height // 2 - eh * (head_dir // 2),
                                                 ew, eh))

            break


def action(action_num):
    global snake_tail_height, snake_tail_width, snake_head_height, snake_head_width, apple_width, apple_height, snake_length, reset_game_flag
    head_dir = (-map[snake_head_height][snake_head_width][1] + action_num) % 4 + 1
    map[snake_head_height][snake_head_width][1] = head_dir
    snake_head_width += dir_width[head_dir]
    snake_head_height += dir_height[head_dir]

    # boundary condition
    if boundary_condition_check(snake_head_width, snake_head_height):
        reset_game_flag = True
        return

    # if snake eat apple, generate new apple
    if map[snake_head_height][snake_head_width][0] == 1:
        while True:
            snake_length += 1
            apple_width = random.randint(0, w - 1)
            apple_height = random.randint(0, h - 1)
            if map[apple_height][apple_width] == [0, 0]:
                map[apple_height][apple_width] = [1, 0]
                break
    else:
        tail_dir = map[snake_tail_height][snake_tail_width][1]
        map[snake_tail_height][snake_tail_width] = [0, 0]
        snake_tail_width += dir_width[tail_dir]
        snake_tail_height += dir_height[tail_dir]

    map[snake_head_height][snake_head_width] = [2, -((head_dir + 1) % 4 + 1)]


def boundary_condition_check(x, y):
    if x < 0 or x >= w:     return True
    if y < 0 or y >= h:     return True
    if map[y][x][0] == 2:   return True
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", dest="width", action="store")
    parser.add_argument("--height", dest="height", action="store")
    args = parser.parse_args()

    w = int(args.width)
    h = int(args.height)
    tile_width, tile_height = 50, 50
    snake_width, snake_height = 30, 30
    connect_width, connect_height = tile_width - snake_width, tile_height - snake_height
    # 0 stop, 1 north, 2 east, 3 south, 4 west
    dir_width = [0, 0, 1, 0, -1]
    dir_height = [0, -1, 0, 1, 0]

    # Initialize the pygame
    pygame.init()

    # create the screen
    screen = pygame.display.set_mode((w * tile_width, h * tile_height))

    # Title and Icon (flaticon.com)
    pygame.display.set_caption("snake_game")
    icon = pygame.image.load('../resource/snake_logo.png')
    pygame.display.set_icon(icon)

    # Img setting
    appleImg = pygame.image.load('../resource/apple.png')
    appleImg = pygame.transform.scale(appleImg, (tile_width, tile_height))

    reset_game()

    # Game Loop
    running = True
    while running:
        reset_game_flag = False

        # RGB
        screen.fill((0, 0, 0))
        draw_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # if keystroke is pressed check
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    print("left")
                    action(0)
                if event.key == pygame.K_UP:
                    print("up")
                    action(1)
                if event.key == pygame.K_RIGHT:
                    print("right")
                    action(2)

        if reset_game_flag: reset_game()

        # update screen
        generate_apple()
        draw_snake()
        pygame.display.update()
