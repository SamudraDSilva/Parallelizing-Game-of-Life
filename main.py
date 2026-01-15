import pygame
import numpy as np
from numba import njit, prange, set_num_threads
import time
import random
import psutil
import os

pygame.init()

# View window size (what you see on screen)
VIEW_WIDTH, VIEW_HEIGHT = 1200, 800

# Grid world size (the actual simulation grid)
GRID_WORLD_WIDTH = 8000
GRID_WORLD_HEIGHT = 4000
CELL_SIZE = 10
GRID_COLUMNS = GRID_WORLD_WIDTH // CELL_SIZE    
GRID_ROWS = GRID_WORLD_HEIGHT // CELL_SIZE      
FPS = 60

screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
clock = pygame.time.Clock()

# Colors
GRAY = (125, 125, 125)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Camera/View system
camera_x = 0
camera_y = 0
zoom = 1.0
MIN_ZOOM = 0.1
MAX_ZOOM = 3.0

# Pan system
is_panning = False
pan_start_x = 0
pan_start_y = 0

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

# Drawing with Pan & Zoom
show_core_colors = False
current_mode = "parallel"

def screen_to_world(screen_x, screen_y):
    """Convert screen coordinates to world (grid) coordinates"""
    world_x = (screen_x / zoom) + camera_x
    world_y = (screen_y / zoom) + camera_y
    return world_x, world_y

def world_to_screen(world_x, world_y):
    """Convert world coordinates to screen coordinates"""
    screen_x = (world_x - camera_x) * zoom
    screen_y = (world_y - camera_y) * zoom
    return screen_x, screen_y

def draw_grid():
    screen.fill(GRAY)
    
    # Calculate visible grid bounds
    view_world_width = VIEW_WIDTH / zoom
    view_world_height = VIEW_HEIGHT / zoom
    
    # Grid cell bounds (which cells are visible)
    start_col = max(0, int(camera_x / CELL_SIZE))
    end_col = min(GRID_COLUMNS, int((camera_x + view_world_width) / CELL_SIZE) + 1)
    start_row = max(0, int(camera_y / CELL_SIZE))
    end_row = min(GRID_ROWS, int((camera_y + view_world_height) / CELL_SIZE) + 1)
    
    # Draw only visible cells
    live_rows, live_cols = np.where(current_generation[start_row:end_row, start_col:end_col])
    live_rows += start_row
    live_cols += start_col
    
    if show_core_colors and current_mode == "parallel":
        core_colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0),
            (255,0,255), (0,255,255), (255,100,0), (100,255,100)
        ]
        num_colors = len(core_colors)
        
        for r, c in zip(live_rows, live_cols):
            world_x = c * CELL_SIZE
            world_y = r * CELL_SIZE
            screen_x, screen_y = world_to_screen(world_x, world_y)
            cell_screen_size = CELL_SIZE * zoom
            
            color_id = r % num_colors
            color = core_colors[color_id]
            pygame.draw.rect(screen, color,
                           (screen_x, screen_y, cell_screen_size, cell_screen_size))
    else:
        for r, c in zip(live_rows, live_cols):
            world_x = c * CELL_SIZE
            world_y = r * CELL_SIZE
            screen_x, screen_y = world_to_screen(world_x, world_y)
            cell_screen_size = CELL_SIZE * zoom
            pygame.draw.rect(screen, YELLOW,
                           (screen_x, screen_y, cell_screen_size, cell_screen_size))
    
    # Draw grid lines (only visible ones)
    cell_screen_size = CELL_SIZE * zoom
    if cell_screen_size > 2:  # Only draw grid if cells are large enough
        for i in range(start_row, end_row + 1):
            world_y = i * CELL_SIZE
            screen_x1, screen_y = world_to_screen(0, world_y)
            screen_x2, _ = world_to_screen(GRID_WORLD_WIDTH, world_y)
            pygame.draw.line(screen, BLACK, (screen_x1, screen_y), (screen_x2, screen_y))
        
        for i in range(start_col, end_col + 1):
            world_x = i * CELL_SIZE
            screen_x, screen_y1 = world_to_screen(world_x, 0)
            _, screen_y2 = world_to_screen(world_x, GRID_WORLD_HEIGHT)
            pygame.draw.line(screen, BLACK, (screen_x, screen_y1), (screen_x, screen_y2))
    
    # Draw UI overlay
    font = pygame.font.Font(None, 24)
    zoom_text = font.render(f"Zoom: {zoom:.1f}x", True, WHITE)
    pos_text = font.render(f"Pos: ({int(camera_x)}, {int(camera_y)})", True, WHITE)
    
    # Black background for text
    pygame.draw.rect(screen, BLACK, (5, 5, 200, 50))
    screen.blit(zoom_text, (10, 10))
    screen.blit(pos_text, (10, 30))

def main():
    global current_generation, current_mode, show_core_colors
    global current_threads, camera_x, camera_y, zoom
    global is_panning, pan_start_x, pan_start_y
    
    running = True
    simulation_running = False
    
    generations_this_second = 0
    last_time = time.time()
    gps = 0.0
    
    print("\n=== Game of Life - Pan & Zoom Edition ===")
    print("Controls:")
    print("   S → Serial mode (1 thread)")
    print("   P → Parallel mode")
    print("   1-9 → Set parallel threads")
    print("   V → Toggle visualizer")
    print("   SPACE → Play / Pause")
    print("   G → Random pattern")
    print("   C → Clear")
    print("   Mouse Wheel → Zoom in/out")
    print("   Middle Mouse / Right Click + Drag → Pan")
    print("   Arrow Keys → Pan")
    print("   Left Click → Toggle cell\n")

    while running:
        clock.tick(FPS)
        
        if simulation_running:
            if current_mode == "serial":
                update_serial()
            else:
                current_generation[:] = update_parallel(current_generation)
            generations_this_second += 1
        
        # GPS calculation
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
        
        # Handle keyboard panning
        keys = pygame.key.get_pressed()
        pan_speed = 20 / zoom
        if keys[pygame.K_LEFT]:
            camera_x -= pan_speed
        if keys[pygame.K_RIGHT]:
            camera_x += pan_speed
        if keys[pygame.K_UP]:
            camera_y -= pan_speed
        if keys[pygame.K_DOWN]:
            camera_y += pan_speed
        
        # Clamp camera
        max_camera_x = GRID_WORLD_WIDTH - VIEW_WIDTH / zoom
        max_camera_y = GRID_WORLD_HEIGHT - VIEW_HEIGHT / zoom
        camera_x = max(0, min(camera_x, max_camera_x))
        camera_y = max(0, min(camera_y, max_camera_y))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom towards mouse position
                mx, my = pygame.mouse.get_pos()
                world_x_before, world_y_before = screen_to_world(mx, my)
                
                zoom_factor = 1.1 if event.y > 0 else 0.9
                zoom *= zoom_factor
                zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
                
                # Adjust camera to keep mouse position steady
                world_x_after, world_y_after = screen_to_world(mx, my)
                camera_x += world_x_before - world_x_after
                camera_y += world_y_before - world_y_after
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2 or event.button == 3:  # Middle or right click
                    is_panning = True
                    pan_start_x, pan_start_y = event.pos
                elif event.button == 1:  # Left click - toggle cell
                    if not is_panning:
                        mx, my = event.pos
                        world_x, world_y = screen_to_world(mx, my)
                        col = int(world_x // CELL_SIZE)
                        row = int(world_y // CELL_SIZE)
                        if 0 <= col < GRID_COLUMNS and 0 <= row < GRID_ROWS:
                            current_generation[row, col] = not current_generation[row, col]
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2 or event.button == 3:
                    is_panning = False
            
            elif event.type == pygame.MOUSEMOTION:
                if is_panning:
                    dx = event.pos[0] - pan_start_x
                    dy = event.pos[1] - pan_start_y
                    camera_x -= dx / zoom
                    camera_y -= dy / zoom
                    pan_start_x, pan_start_y = event.pos
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_running = not simulation_running
                elif event.key == pygame.K_c:
                    current_generation[:] = False
                elif event.key == pygame.K_g:
                    num = random.randint(10000, 15000)
                    current_generation[:] = False
                    indices = random.sample(range(GRID_ROWS * GRID_COLUMNS), num)
                    current_generation.flat[indices] = True
                elif event.key == pygame.K_v:
                    show_core_colors = not show_core_colors
                elif event.key == pygame.K_s:
                    current_mode = "serial"
                elif event.key == pygame.K_p:
                    current_mode = "parallel"
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    requested = event.key - pygame.K_0
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