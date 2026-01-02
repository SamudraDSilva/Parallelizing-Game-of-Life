import pygame
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
import time
import random
import psutil
import os

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 8000, 400
CELL_SIZE = 10
GRID_COLUMNS = WINDOW_WIDTH // CELL_SIZE    
GRID_ROWS = WINDOW_HEIGHT // CELL_SIZE      
FPS = 60

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()


GRAY = (125, 125, 125)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)


current_generation = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=np.bool_)

# Parallel Update
@njit(parallel=True)
def update_parallel(grid):
    height, width = grid.shape
    next_gen = np.zeros((height, width), dtype=np.bool_)
    
    for row in prange(height):
        for col in range(width):
            live_neighbors = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr = row + dr
                    nc = col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr, nc]:
                            live_neighbors += 1
            
            alive = grid[row, col]
            next_gen[row, col] = (live_neighbors == 3) or (alive and live_neighbors == 2)
    
    return next_gen

# Warm-up Numba
print("Warming up Numba compiler...")
update_parallel(current_generation.copy())
print("Ready!")

# Default: use all available threads
max_threads = os.cpu_count()
current_threads = max_threads
set_num_threads(current_threads)
print(f"Numba using {current_threads} threads (max: {max_threads})")

# Serial Update 
def update_serial():
    global current_generation
    height, width = current_generation.shape
    next_gen = np.zeros_like(current_generation)
    
    for row in range(height):
        for col in range(width):
            live_neighbors = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        live_neighbors += current_generation[nr, nc]
            
            alive = current_generation[row, col]
            next_gen[row, col] = (live_neighbors == 3) or (alive and live_neighbors == 2)
    
    current_generation[:] = next_gen

# Drawing with Visualizer 
show_core_colors = False
current_mode = "parallel"  # Start in parallel with full threads

def draw_grid():
    screen.fill(GRAY)
    
    live_rows, live_cols = np.where(current_generation)
    
    if show_core_colors and current_mode == "parallel":
        core_colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0),
            (255,0,255), (0,255,255), (255,100,0), (100,255,100)
        ]
        num_colors = len(core_colors)
        
        for r, c in zip(live_rows, live_cols):
            color_id = r % num_colors
            color = core_colors[color_id]
            pygame.draw.rect(screen, color,
                             (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    else:
        for r, c in zip(live_rows, live_cols):
            pygame.draw.rect(screen, YELLOW,
                             (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Grid lines
    for i in range(GRID_ROWS + 1):
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (WINDOW_WIDTH, i * CELL_SIZE))
    for i in range(GRID_COLUMNS + 1):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_HEIGHT))

def main():
    global current_generation, current_mode, show_core_colors
    global current_threads
    
    running = True
    simulation_running = False
    
    generations_this_second = 0
    last_time = time.time()
    gps = 0.0
    
    print("\n=== Game of Life - Custom Thread Control ===")
    print("Controls:")
    print("   S → Serial mode (1 thread)")
    print("   P → Parallel mode")
    print("   1-9 → Set parallel threads (1 to 9, or max if > available)")
    print("   V → Toggle visualizer")
    print("   SPACE → Play / Pause")
    print("   G → Random pattern")
    print("   C → Clear")
    print("   Click → Toggle cell\n")

    while running:
        clock.tick(FPS)
        
        if simulation_running:
            if current_mode == "serial":
                update_serial()
            else:
                current_generation[:] = update_parallel(current_generation)
            generations_this_second += 1
        
        # GPS
        now = time.time()
        if now - last_time >= 1.0:
            gps = generations_this_second / (now - last_time)
            generations_this_second = 0
            last_time = now
        
        cpu_usage = psutil.cpu_percent(interval=None)
        live_cells = np.count_nonzero(current_generation)
        vis_status = "ON" if show_core_colors else "OFF"
        thread_info = f"Threads: {current_threads}/{max_threads}" if current_mode == "parallel" else "Serial"
        
        pygame.display.set_caption(
            f"Mode: {current_mode.upper()} | {thread_info} | "
            f"GPS: {gps:.0f} | Live: {live_cells} | CPU: {cpu_usage:.0f}% | Vis: {vis_status}"
        )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                col = mx // CELL_SIZE
                row = my // CELL_SIZE
                if 0 <= col < GRID_COLUMNS and 0 <= row < GRID_ROWS:
                    current_generation[row, col] = not current_generation[row, col]
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_running = not simulation_running
                elif event.key == pygame.K_c:
                    current_generation[:] = False
                elif event.key == pygame.K_g:
                    num = random.randint(4000, 6000)
                    current_generation[:] = False
                    indices = random.sample(range(GRID_ROWS * GRID_COLUMNS), num)
                    current_generation.flat[indices] = True
                elif event.key == pygame.K_v:
                    show_core_colors = not show_core_colors
                elif event.key == pygame.K_s:
                    current_mode = "serial"
                elif event.key == pygame.K_p:
                    current_mode = "parallel"
                # Custom thread control: keys 1 to 9
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    requested = event.key - pygame.K_0  # 1→1, 2→2, ..., 9→9
                    new_threads = min(requested, max_threads)
                    if new_threads != current_threads:
                        current_threads = new_threads
                        set_num_threads(current_threads)
                        print(f"Numba threads set to: {current_threads}")
        
        draw_grid()
        pygame.display.update()
    
    pygame.quit()

if __name__ == "__main__":
    main()