import pygame
import random
import time

pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (125,125,125)
YELLOW = (255,255,0)

WIDTH, HEIGHT = 400, 400
TILE_SIZE = 10 
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE
FPS = 60

screen = pygame.display.set_mode((WIDTH,HEIGHT))

clock = pygame.time.Clock()



def gen(num):
    return set([(random.randrange(0,GRID_HEIGHT),random.randrange(0,GRID_WIDTH)) for _ in range(num)])

def draw_grid(positions):

    for position in positions:
        col, row = position
        top_left = (col * TILE_SIZE, row * TILE_SIZE)
        pygame.draw.rect(screen,YELLOW,(*top_left,TILE_SIZE,TILE_SIZE))

    for row in range(GRID_HEIGHT):
        pygame.draw.line(screen,BLACK, (0, row * TILE_SIZE), (WIDTH, row * TILE_SIZE))
    
    for col in range(GRID_WIDTH):
        pygame.draw.line(screen,BLACK,(col * TILE_SIZE, 0),(col * TILE_SIZE, HEIGHT))

def adjust_grid(positions):
    all_neighbors = set()
    new_position = set()

    for position in positions:
        neighbors = get_neighbors(position)
        all_neighbors.update(neighbors)

        neighbors = list(filter(lambda x: x in positions, neighbors))
        if len(neighbors) in [2,3]:
            new_position.add(position)
        
    for position in all_neighbors:
        neighbors = get_neighbors(position)
        neighbors = list(filter(lambda x: x in positions, neighbors))
        if len(neighbors) == 3:
            new_position.add(position)

    return new_position

def get_neighbors(pos):
    x, y = pos
    neighbors = []
    for dx in [-1,0,1]:
        if x + dx < 0 or x + dx >= GRID_WIDTH:
            continue
        for dy in [-1,0,1]:
            if y + dy < 0 or y + dy >= GRID_HEIGHT:
                continue
            if dx == 0 and dy == 0:
                continue
            neighbors.append((x + dx, y + dy))

    return neighbors

def main():
    running = True
    playing = False
    positions = set()

    count = 0
    update_freq = 120

    # Performance measument variables
    generation_count = 0
    gps = 0.0
    last_gps_time = time.time()

    while running:
        clock.tick(FPS)

        if playing: 
            count += 1
        
        if count >= update_freq:
            count = 0
            positions = adjust_grid(positions)

            # Count this for generation for GPS calculation
            generation_count +=1
            current_time = time.time()

            if current_time - last_gps_time >= 1.0:
                gps = generation_count / (current_time - last_gps_time)
                generation_count = 0
                last_gps_time = current_time
                 

        status = "Playing" if playing else "Pause"
        pop = len(positions)
        pygame.display.set_caption(f"Conway's Game of Life | GPS: {gps:.1f} | Population: {pop}| {status}")


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // TILE_SIZE
                row = y // TILE_SIZE
                pos = (col,row)
            
                if pos in positions:
                    positions.remove(pos)
                else:
                    positions.add(pos)
            
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    playing = not playing

                if event.key == pygame.K_c:
                    positions = set()
                    playing = False
                
                if event.key == pygame.K_g:
                    positions = gen(random.randrange(2,5) * GRID_WIDTH)

        
        screen.fill(GRAY)
        draw_grid(positions)
        pygame.display.update()
        
    pygame.quit()

if __name__ == "__main__":
    main()
