#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <future>

#define size 20000

double **matrix = new double*[size];
double *vec = new double[size];
double *res = new double[size];

void initialization()
{
    for(int i = 0; i < size; i++) { matrix[i] = new double[size]; }

    for (int i = 0; i < size; i++)
    {
        vec[i] = size + 1;
        res[i] = 0;
        
        for (int j = 0; j < size; j++)
        {
            if (i == j) { matrix[i][j] = 2.0; }
            else { matrix[i][j] = 1.0; }
        }
    }
}

void thread_work(int thread_id, int num_threads)
{
    int new_size = size / num_threads;

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

int main(int argc, char* argv[])
{
    auto init_future = std::async(std::launch::async, initialization);

    unsigned int num_threads = std::stoi(argv[1]);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    init_future.get();
    
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(thread_work, i, num_threads);
    }

    for (auto& t : threads) { if (t.joinable()) { t.join(); } }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Parallel time: " << duration.count() << " ms\n";

    for(int i = 0; i < size; i++) { delete[] matrix[i]; }
    delete[] matrix;
    delete[] vec;
    delete[] res;
    return 0;
}