#include <boost/program_options.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace po = boost::program_options;

#define IDX(i, j, n) ((i) * (n) + (j))

// последовательное выполнение в gpu-блоке
#pragma acc routine seq
static inline double lerp(double a, double b, double t) { return a + (b - a) * t; }

// инициализация границ
void init_boundaries(double* grid, double* newgrid, int n,
                     double top_left, double top_right,
                     double bottom_right, double bottom_left)
{
    // распараллелить на gpu
    #pragma acc loop
    for (int j = 0; j < n; j++)
    {
        double t = static_cast<double>(j) / (n - 1);
        double top = lerp(top_left, top_right, t);
        double bottom = lerp(bottom_left, bottom_right, t);
        grid[IDX(0, j, n)] = newgrid[IDX(0, j, n)] = top;
        grid[IDX(n - 1, j, n)] = newgrid[IDX(n - 1, j, n)] = bottom;
    }

    #pragma acc loop
    for (int i = 0; i < n; i++)
    {
        double t = static_cast<double>(i) / (n - 1);
        double left = lerp(top_left, bottom_left, t);
        double right = lerp(top_right, bottom_right, t);
        grid[IDX(i, 0, n)] = newgrid[IDX(i, 0, n)] = left;
        grid[IDX(i, n - 1, n)] = newgrid[IDX(i, n - 1, n)] = right;
    }

    grid[IDX(0, 0, n)] = newgrid[IDX(0, 0, n)] = top_left;
    grid[IDX(0, n - 1, n)] = newgrid[IDX(0, n - 1, n)] = top_right;
    grid[IDX(n - 1, n - 1, n)] = newgrid[IDX(n - 1, n - 1, n)] = bottom_right;
    grid[IDX(n - 1, 0, n)] = newgrid[IDX(n - 1, 0, n)] = bottom_left;

    // объединяет два вложенных цикла в один параллельный цикл
    #pragma acc loop collapse(2)
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 1; j < n - 1; j++)
        {
            int id = IDX(i, j, n);
            grid[id] = 0.0;
            newgrid[id] = 0.0;
        }
    }
}

int main(int argc, char* argv[])
{
    int n;
    int max_iters;
    double eps;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "show help message")
        ("size,n", po::value<int>(&n)->default_value(128), "grid size N for NxN")
        ("eps,e", po::value<double>(&eps)->default_value(1e-6), "accuracy threshold")
        ("max-iters,i", po::value<int>(&max_iters)->default_value(1000000), "maximum iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    const double top_left = 10.0;
    const double top_right = 20.0;
    const double bottom_right = 30.0;
    const double bottom_left = 20.0;

    double* grid = new double[n * n];
    double* newgrid = new double[n * n];
    double maxdiff = 0.0;
    int niters = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // выделение памяти на GPU и копирование данных в начале блока
    #pragma acc data create(grid[0:n*n], newgrid[0:n*n])
    {
        #pragma acc parallel present(grid, newgrid)
        {
            init_boundaries(grid, newgrid, n, top_left, top_right, bottom_right, bottom_left);
        }
        
        for (; niters < max_iters; niters++)
        {
            // параллельное выполнение с объединением циклов
            #pragma acc parallel loop collapse(2) present(grid[0:n*n], newgrid[0:n*n])
            for (int i = 1; i < n - 1; i++)
            {
                for (int j = 1; j < n - 1; j++)
                {
                    int id = i * n + j;
                    double up    = grid[id - n];
                    double down  = grid[id + n];
                    double left  = grid[id - 1];
                    double right = grid[id + 1];
                    newgrid[id] = 0.25 * (up + down + left + right);
                }
            }

            maxdiff = 0.0;
            // + локал копия maxdiff, чтобы избежать гонки данных
            #pragma acc parallel loop collapse(2) reduction(max:maxdiff) present(grid[0:n*n], newgrid[0:n*n])
            for (int i = 1; i < n - 1; i++)
            {
                for (int j = 1; j < n - 1; j++)
                {
                    int id = i * n + j;
                    double diff = newgrid[id] - grid[id];
                    if (diff < 0.0) diff = -diff;
                    if (diff > maxdiff) maxdiff = diff;
                }
            }

            double* tmp = grid;
            grid = newgrid;
            newgrid = tmp;

            if (maxdiff < eps)
            {
                niters++;
                break;
            }
        }

        // перенос данных на cpu
        #pragma acc update self(grid[0:n*n])
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double seconds = duration.count() / 1000.0;

    std::cout << "GPU: \n";
    std::cout << "Iterations     : " << niters << "\n";
    std::cout << "Reached error  : " << std::scientific << maxdiff << "\n";
    std::cout << "Execution time : " << std::fixed << std::setprecision(3) << seconds << " seconds (" << duration.count() << " ms)\n";

    delete[] grid;
    delete[] newgrid;
    return 0;
}