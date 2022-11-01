import pygame
import argparse
import random


def reset_game():
    global map, snake_head_width, snake_head_height, snake_tail_width, snake_tail_height, snake_length
    global apple_width, apple_height, wall_width, wall_height

    map = [[[0, 0] for __ in range(w)] for _ in range(h)]

    # snake
    snake_head_width, snake_head_height = 2, h // 2 - 1
    snake_tail_width, snake_tail_height = 1, h // 2 - 1
    map[snake_head_height][snake_head_width] = [2, 2]  # head
    map[snake_tail_height][snake_tail_width] = [2, 2]  # tail
    snake_length = 2

    # apple
    apple_width, apple_height = [w - 2], [h // 2 - 1]
    map[h // 2 - 1][w - 2] = [1, 0]
    for i in range(apple_num - 1):
        apple_width_tmp, apple_height_tmp = random_empty_width_height_apple()
        map[apple_height_tmp][apple_width_tmp] = [1, 0]  # food
        apple_width.append(apple_width_tmp)
        apple_height.append(apple_height_tmp)

    # wall
    wall_width, wall_height = list(), list()


def draw_screen():
    GREEN1, GREEN2 = (170, 215, 81), (162, 209, 73)
    screen.fill(GREEN1)
    for col in range(w):
        for row in range(col % 2, h, 2):
            pygame.draw.rect(screen, GREEN2, (col * tile_width, row * tile_height, tile_width, tile_height))


def draw_apple():
    # draw on screen
    for i in range(apple_num):
        screen.blit(appleImg, (apple_width[i] * tile_width, apple_height[i] * tile_height))


def draw_snake():
    BLUE = (71, 117, 235)
    BLACK = (0, 0, 0)
    draw_width = snake_tail_width
    draw_height = snake_tail_height
    while True:
        dir = map[draw_height][draw_width][1]
        if dir == 1:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width + connect_width // 2, draw_height * tile_height - connect_height // 2,
                snake_width, snake_height + connect_height))
        if dir == 2:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2,
                snake_width + connect_width, snake_height))
        if dir == 3:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2,
                snake_width, snake_height + connect_height))
        if dir == 4:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width - connect_width // 2, draw_height * tile_height + connect_height // 2,
                snake_width + connect_width, snake_height))
        draw_width = draw_width + dir_width[dir]
        draw_height = draw_height + dir_height[dir]
        if draw_height == snake_head_height and draw_width == snake_head_width:
            pygame.draw.rect(screen, BLUE, (
                draw_width * tile_width + connect_width // 2, draw_height * tile_height + connect_height // 2,
                snake_width,
                snake_height))

            # snake eyes
            eh, ew = 9, 6
            head_dir = map[draw_height][draw_width][1]
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


def draw_wall():
    small_space = 5
    BLACK = (0, 0, 0)
    for i in range(len(wall_width)):
        pygame.draw.rect(screen, BLACK, (
            wall_width[i] * tile_width + small_space, wall_height[i] * tile_height + small_space,
            tile_width - small_space * 2, tile_height - small_space * 2))


def action(action_num):
    global snake_tail_height, snake_tail_width, snake_head_height, snake_head_width
    global apple_width, apple_height, snake_length, reset_game_flag
    global wall_width, wall_height

    head_dir = (map[snake_head_height][snake_head_width][1] + action_num - 2) % 4 + 1
    map[snake_head_height][snake_head_width][1] = head_dir
    snake_head_width += dir_width[head_dir]
    snake_head_height += dir_height[head_dir]

    # boundary condition
    if boundary_condition_check(snake_head_width, snake_head_height):
        reset_game_flag = True
        return

    # if snake eat apple, generate new apple
    if map[snake_head_height][snake_head_width][0] == 1:
        for i in range(apple_num):
            if (apple_width[i] == snake_head_width and apple_height[i] == snake_head_height):
                apple_remove_idx = i

        snake_length += 1
        apple_width[apple_remove_idx], apple_height[apple_remove_idx] = random_empty_width_height_apple()
        map[apple_height[apple_remove_idx]][apple_width[apple_remove_idx]] = [1, 0]

        if wall == 1 and snake_length % wall_freqency == 0 and (len(wall_width) < w * h // 10):
            wall_width_tmp, wall_height_tmp = random_empty_width_height_wall()
            map[wall_height_tmp][wall_width_tmp] = [3, 0]
            wall_width.append(wall_width_tmp)
            wall_height.append(wall_height_tmp)

    else:
        tail_dir = map[snake_tail_height][snake_tail_width][1]
        map[snake_tail_height][snake_tail_width] = [0, 0]
        snake_tail_width += dir_width[tail_dir]
        snake_tail_height += dir_height[tail_dir]

    map[snake_head_height][snake_head_width] = [2, head_dir]


def random_empty_width_height_apple():
    while True:
        W = random.randint(0, w - 1)
        H = random.randint(0, h - 1)
        if map[H][W] == [0, 0]:  return W, H


# 3 * 3 tile -> one wall
def random_empty_width_height_wall():
    while True:
        flag = 0
        pw = [-1, 0, 1, 1, 1, 0, -1, -1]
        ph = [-1, -1, -1, 0, 1, 1, 1, 0]
        W = random.randint(0, w - 1)
        H = random.randint(0, h - 1)
        if map[H][W] != [0, 0]:  continue
        for i in range(8):
            if H + ph[i] < 0 or H + ph[i] >= h: continue
            if W + pw[i] < 0 or W + pw[i] >= w: continue
            if map[H + ph[i]][W + pw[i]][0] == 3:   flag = 1

        if not flag:  return W, H


def boundary_condition_check(x, y):
    if x < 0 or x >= w:     return True
    if y < 0 or y >= h:     return True
    if map[y][x][0] == 2:   return True
    if map[y][x][0] == 3:   return True
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", dest="width", action="store", default=15)
    parser.add_argument("--height", dest="height", action="store", default=10)
    # define number of apples
    parser.add_argument("--apple", dest="apple", action="store", default=5)
    # 0 -> basic environment
    # 1 -> generate wall if snake eats food
    parser.add_argument("--setting", dest="setting", action="store", default=1)
    args = parser.parse_args()

    w = int(args.width)
    h = int(args.height)
    apple_num = int(args.apple)
    wall = int(args.setting)

    tile_width, tile_height = 50, 50
    snake_width, snake_height = 30, 30
    connect_width, connect_height = tile_width - snake_width, tile_height - snake_height
    # 0 stop, 1 north, 2 east, 3 south, 4 west
    dir_width = [0, 0, 1, 0, -1]
    dir_height = [0, -1, 0, 1, 0]
    # if setting is 1
    wall_freqency = 2

    # Initialize the pygame
    pygame.init()

    # create the screen
    screen = pygame.display.set_mode((w * tile_width, h * tile_height))

    # Title and Icon (flaticon.com)
    pygame.display.set_caption("snake_game")
    icon = pygame.image.load('snake_logo.png')
    pygame.display.set_icon(icon)

    # Img setting
    appleImg = pygame.image.load('apple.png')
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
        draw_apple()
        draw_snake()
        if wall == 1:    draw_wall()
        pygame.display.update()
