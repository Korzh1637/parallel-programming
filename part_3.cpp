#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <cmath>
#define size 40000


int main()
{
    omp_set_num_threads(1);

    double **matrix = new double*[size];
    for(int i = 0; i < size; i++) { matrix[i] = new double[size]; }

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (i == j) { matrix[i][j] = 2.0; }
            else { matrix[i][j] = 1.0; }
        }
    }

    double *vec = new double[size];
    double *res = new double[size];
    double *res_new = new double[size];

    for (int i = 0; i < size; i++)
    {
        vec[i] = size + 1;
        res[i] = 0;
        res_new[i] = 0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int iter;
    double max_diff;
    
    for (iter = 0; iter < 1000; iter++)
    {
        max_diff = 0.0;
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int new_size = size / num_threads;
            int thread_id = omp_get_thread_num();

            int start = thread_id * new_size;
            int finish = thread_id == num_threads - 1 ? size : (thread_id + 1) * new_size;

            for (int i = start; i < finish; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < size; j++)
                {
                    if (i != j) { sum += matrix[i][j] * res[j]; }
                }
                
                double temp_new_res = (vec[i] - sum) / matrix[i][i];
                res_new[i] = (1 - 0.0001) * res[i] + 0.0001 * temp_new_res;
            }

            double thread_max = 0;

            for (int i = start; i < finish; i++)
            {
                double diff = fabs(res_new[i] - res[i]);
                if (diff > thread_max) { thread_max = diff; }
            }

            #pragma omp critical
            if (thread_max > max_diff) { max_diff = thread_max; }
        }

        #pragma omp parallel for
        for (int i = 0; i < size; i++) { res[i] = res_new[i]; }

        // проверка сходимости
        if (max_diff < 1e-6) { break; }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Parallel time: " << duration.count() << " ms\n";

    // std::ofstream file("result.txt", std::ios::out);
    for (int i = 0; i < 10; i++) { std::cout << res[i] << std::endl; }

    for(int i = 0; i < size; i++) { delete[] matrix[i]; }
    delete[] matrix;
    delete[] vec;
    delete[] res;
    delete[] res_new;
    return 0;
}