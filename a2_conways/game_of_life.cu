// list of coords, representing live cells
// for each live cell, create list of neighboring cells
// then, flatten the lists into one list
// sort and group the list such that repeat neighbor cells are together
// now you have a list of cells, and the # it was considered a neighbor by a living cell (and thus, the inverse is the number of dead neighbors)
// (cells not in the list have all dead neighbors)
// <2 live neighbors, alive or dead - Dead
// 2-3 live neighbors & cell is live - Alive
// 2 live neighbors & cell is dead - Dead
// 3 live neighbors & cell is dead - Alive
// 4+ live neighbors - Dead
//
//
//

#include <stdio.h>
#include <cuda.h>
#include <ncurses.h>
#include <stdint.h>
#include <unistd.h>  // for usleep()
#include <stdlib.h>
#include <string.h>

#define DELAY
const int N = 20;
uint8_t *current_grid;
uint8_t *next_grid;
uint8_t *host_grid;

__device__ int count_neighbors(uint8_t* grid, int rows, int cols, int y, int x) {
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0)
                continue;  // skip the center cell

            int ny = (y + dy + rows) % rows;
            int nx = (x + dx + cols) % cols;

            count += grid[ny * cols + nx];
        }
    }

    return count;
}
__global__ void conway_kernel(uint8_t*ng, uint8_t*cg){
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  // for each cell in the row,
  bool still_alive;
  for (int i = 0; i < N; i++){
    // count the neighbors
    int c = count_neighbors(cg, N, N, row, i);

    // and play the game accordingly
    switch (c) {
      case 2: 
        // alive if alive
        still_alive = cg[i + row*N];
        break;
      case 3: 
        // alive
        still_alive = 1;
        break;
      default:
        // dead
        still_alive = 0;
    }

    // update next_grid for that cell
    ng[i + row*N] = still_alive;
  }

};

size_t grid_bytes = N * N * sizeof(uint8_t);

void draw_grid_and_log(uint8_t* grid, int rows, int cols);

int main(int argc, char *argv[]) {
  // NOTE: using cudaMalloc for global Device side memory
  cudaMalloc(&current_grid, grid_bytes);
  cudaMalloc(&next_grid, grid_bytes);


  printf("allocating grid\n");
  // host_grid = (uint8_t*) malloc(grid_bytes);
  // NOTE: using Pinned memory instead of malloc (still Host side memory, but not paged. much faster because DMA)
  // pinned memory useful for stuff like framebuffers, that still need to be host side
  cudaHostAlloc(&host_grid, grid_bytes, cudaHostAllocDefault);

  printf("initializing grid\n");
  memset(host_grid, 0, grid_bytes);

  for (int x = 0; x < N; x++){
    for (int y = 0; y < N; y++){
      host_grid[y*N + x] = rand() % 2; // 0 or 1
    }
  }
  
  //print_grid(host_grid, N, N);
  draw_grid_and_log(host_grid, N,N);
  //print_grid(host_grid, N, N);



  printf("Done\n");
  cudaFree(current_grid);
  cudaFree(next_grid);

};


void run_game(){
  // run the actual game on gpu
  //printf("copying grid to device\n");
  cudaMemcpy(current_grid, host_grid, grid_bytes, cudaMemcpyHostToDevice);
  
  //printf("launching kernel\n");
  cudaError_t err = cudaSuccess;
  conway_kernel<<<1,N>>>(next_grid, current_grid);
  
  err = cudaGetLastError();
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();
  //printf("kernel complete\n");
  //printf("copying next grid from device\n");
  cudaMemcpy(host_grid, next_grid, grid_bytes, cudaMemcpyDeviceToHost);
}

double time_game(){
    struct timespec begin, end;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &begin);
    
    // spawn threads to do work here
    run_game();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    return elapsed;
}


void draw_grid_and_log(uint8_t* grid, int rows, int cols) {
    initscr();              // Start ncurses
    curs_set(0);            // Hide cursor
    noecho();               // Don't echo keypresses
    nodelay(stdscr, TRUE);  // Non-blocking input
    start_color();

    init_pair(1, COLOR_GREEN, COLOR_BLACK); // Alive
    init_pair(2, COLOR_BLACK, COLOR_BLACK); // Dead

    int term_rows, term_cols;
    getmaxyx(stdscr, term_rows, term_cols);

    int log_height = 5;
    int grid_height = 35;
    int window_cols = cols;

    // Create windows
    WINDOW* grid_win = newwin(grid_height, window_cols, 0, 0);
    WINDOW* log_win = newwin(log_height, window_cols, grid_height, 0);

    int generation = 0;
    double elapsed = 0.0;
    double total_elapsed = 0.0;

    while (1) {
        werase(grid_win);
        werase(log_win);

        // Draw grid
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                int idx = y * cols + x;
                if (grid[idx]) {
                    wattron(grid_win, COLOR_PAIR(1));
                    mvwaddch(grid_win, y, x, 'O');
                    wattroff(grid_win, COLOR_PAIR(1));
                } else {
                    wattron(grid_win, COLOR_PAIR(2));
                    mvwaddch(grid_win, y, x, ' ');
                    wattroff(grid_win, COLOR_PAIR(2));
                }
            }
        }

        // Draw log info
        int alive_count = 0;
        for (int i = 0; i < rows * cols; ++i) {
            if (grid[i]) alive_count++;
        }

        mvwprintw(log_win, 0, 0, "Generation: %d", generation);
        mvwprintw(log_win, 1, 0, "Alive cells: %d", alive_count);
        mvwprintw(log_win, 2, 0, "Kernel time: %f", elapsed);
        mvwprintw(log_win, 3, 0, "average time: %f", (double)total_elapsed/generation);
        mvwprintw(log_win, 4, 0, "N = %d", N);

        // Refresh windows
        wrefresh(grid_win);
        wrefresh(log_win);

#ifdef DELAY 
        usleep(100000); // 100ms delay
#endif

        // Exit check
        int ch = getch();
        if (ch == 'q') break;

        // Here you would launch your CUDA kernel and swap buffers
        // For demo purposes, we just increment the generation
        generation++;
        elapsed = time_game();
        total_elapsed += elapsed;
    }

    // Cleanup
    delwin(grid_win);
    delwin(log_win);
    endwin();
}



