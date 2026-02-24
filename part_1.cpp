#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <chrono>
#define size 20000

int main()
{
    double **matrix = new double*[size];
    for(int i = 0; i < size; i++) { matrix[i] = new double[size]; }

    double *vec = new double[size];
    double *res = new double[size];

    for (int i = 0; i < size; i++)
    {
        vec[i] = size + 1;
        res[i] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                matrix[i][j] = 2.0;
            else
                matrix[i][j] = 1.0;
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int new_size = size / num_threads;
        int thread_id = omp_get_thread_num();

        int start = thread_id * new_size;
        int finish = thread_id == num_threads - 1 ? size : (thread_id + 1) * new_size;

        for (int i = start; i < finish; i++)
        {
            for (int j = 0; j < size; j++)
            {
                res[i] += matrix[i][j] * vec[j];
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Parallel time: " << duration.count() << " ms\n";

    for(int i = 0; i < size; i++) { delete[] matrix[i]; }
    delete[] matrix;
    delete[] vec;
    delete[] res;
    return 0;
}