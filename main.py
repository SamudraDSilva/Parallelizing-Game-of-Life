import pygame
import numpy as np
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

grid = np.zeros((GRID_HEIGHT,GRID_WIDTH),dtype=bool) # Alive = True, Dead = False

def gen(num):
    grid[:] = False
    if num > GRID_WIDTH * GRID_HEIGHT:
        num = GRID_WIDTH * GRID_HEIGHT
    indices = random.sample(range(GRID_WIDTH * GRID_HEIGHT), num)
    grid.flat[indices] = True

def draw_grid():

    # Draw Live Cells
    live_rows, live_cols = np.where(grid)
    for row, col in zip(live_rows,live_cols):
        top_left = (col * TILE_SIZE, row * TILE_SIZE)
        pygame.draw.rect(screen,YELLOW,(*top_left,TILE_SIZE,TILE_SIZE))

    for row in range(GRID_HEIGHT):
        pygame.draw.line(screen,BLACK, (0, row * TILE_SIZE), (WIDTH, row * TILE_SIZE))
    
    for col in range(GRID_WIDTH):
        pygame.draw.line(screen,BLACK,(col * TILE_SIZE, 0),(col * TILE_SIZE, HEIGHT))

# ------------------- Serial Update Function -------------------

def update_grid_serial():
    global grid
    new_grid = np.zeros_like(grid)

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            total = 0
            for dj in [-1,0,1]:
                for di in [-1,0,1]:
                    if dj == 0 and di == 0:
                        continue
                    ni , nj = i + di, j + dj
                    if 0 <= ni < GRID_HEIGHT and 0 <= nj < GRID_WIDTH:
                        total += grid[ni,nj]
        
            if grid[i,j]:
                if total == 2 or total == 3:
                    new_grid[i, j] = True
            else:
                if total == 3:
                    new_grid[i, j] = True
    
    grid = new_grid


def main():
    global grid
    running = True
    playing = False

    count = 0
    update_freq = 30

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
            update_grid_serial()

            # Count this for generation for GPS calculation
            generation_count +=1
            current_time = time.time()

            if current_time - last_gps_time >= 1.0:
                gps = generation_count / (current_time - last_gps_time)
                generation_count = 0
                last_gps_time = current_time
                 

        status = "Playing" if playing else "Pause"
        pop = np.count_nonzero(grid)
        pygame.display.set_caption(f"Conway's Game of Life | GPS: {gps:.1f} | Population: {pop}| {status}")


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // TILE_SIZE
                row = y // TILE_SIZE
                pos = (col,row)
            
                if 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT:
                    grid[row,col] = not grid[row,col]
            
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    playing = not playing

                if event.key == pygame.K_c:
                    grid[:] = False
                    playing = False
                
                if event.key == pygame.K_g:
                    gen(random.randint(300,800))
        
            
        screen.fill(GRAY)
        draw_grid()
        pygame.display.update()
        
    pygame.quit()

if __name__ == "__main__":
    main()
