#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <cmath>
#define nstep 80000000

double func(double x) { return exp(-x*x); }

double integrate_omp(double (*func)(double), double a, double b)
{
    double h = (b - a) / nstep;
    double sum = 0.0;
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int new_size = nstep / num_threads;
        int thread_id = omp_get_thread_num();

        int start = thread_id * new_size;
        int finish = thread_id == num_threads - 1 ? nstep : (thread_id + 1) * new_size;

        double local_sum = 0;

        for (int i = start; i < finish; i++)
        {
            local_sum += func(a + h * (i + 0.5));
        }

        #pragma omp atomic
        sum += local_sum;
    }
    sum *= h;
    return sum;
}

int main()
{
    double a = -4.0;
    double b = 4.0;

    auto start = std::chrono::high_resolution_clock::now();

    integrate_omp(func, a, b);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel time: " << duration.count() << " ms\n";

    // std::ofstream file("result.txt", std::ios::out);
    // std::cout << res << std::endl;

    // delete[] vec1;
    return 0;
}