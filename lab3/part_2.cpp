#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <cmath>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <regex>

template<typename T>
T func_sin(T x) { return std::sin(x); }

template<typename T>
T func_sqrt(T x) { return std::sqrt(x); }

template<typename T>
T func_pow(T x, T y) { return std::pow(x, y); }

template<typename T>
class Server
{
    std::queue<std::pair<size_t, std::packaged_task<T()>>> tasks; // очередь задач
    size_t next_id = 0;                                           // id задач
    std::condition_variable cond_var;                             // ожидание и пробуждение потоков
    std::mutex mut;                                               // мьютекс
    std::unordered_map<size_t, T> results;                        // готовые результаты
    std::vector<std::jthread> workers;                            // потоки
    std::atomic<bool> stop_flag{false};
    
    void process_tasks(std::stop_token stoken)
    {
        std::unique_lock<std::mutex> lock{mut, std::defer_lock};

        while (!stoken.stop_requested() && !stop_flag)
        {
            std::unique_lock<std::mutex> lock(mut);
            
            cond_var.wait(lock, [this, &stoken] { return !tasks.empty() || stoken.stop_requested() || stop_flag; });
            
            if (stop_flag || stoken.stop_requested()) { break; }

            if (!tasks.empty())
            {
                auto [id, task] = std::move(tasks.front());
                tasks.pop();
                auto future = task.get_future();
                lock.unlock();
                
                task();
                T result = future.get();
                lock.lock();
                results[id] = result;
            }
        }
    }
    
    public:
    ~Server() { stop(); }

    void start(int num_threads = std::thread::hardware_concurrency())
    {
        if (num_threads == 0) num_threads = 2;
        if (num_threads > 16) num_threads = 16;
        
        workers.clear();
        stop_flag = false;
        
        for (int i = 0; i < num_threads; i++)
        {
            workers.emplace_back([this](std::stop_token stoken) { process_tasks(stoken); });
        }
        
        std::cout << "Thread Pool started with " << num_threads << " threads\n";
    }
    
    void stop()
    {
        stop_flag = true;
        cond_var.notify_all();
        
        for (auto& worker : workers) { worker.request_stop(); }
        workers.clear();
    }
    
    size_t add_task(std::function<T()> task)
    {
        std::packaged_task<T()> packaged_task(task);
        size_t id = next_id++;
        
        {
            std::lock_guard<std::mutex> lock(mut);
            tasks.push({id, std::move(packaged_task)});
        }
        cond_var.notify_one();
        return id;
    }
    
    T request_result(size_t id_res)
    {
        std::unique_lock<std::mutex> lock(mut);
        cond_var.wait(lock, [this, id_res] { return results.find(id_res) != results.end(); });
        return results[id_res];
    }
};

void add_task_sin(Server<double>& server, int N, int client_id)
{
    std::ofstream file("client_sin.txt");
    std::vector<size_t> ids;
    std::vector<double> args;
    
    for (int i = 0; i < N; i++)
    {
        double arg = ((double)rand() / RAND_MAX) * 1000;
        std::function<double()> task = std::bind(func_sin<double>, arg);
        ids.push_back(server.add_task(task));
        args.push_back(arg);
    }
    
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < ids.size(); i++)
    {
        double res = server.request_result(ids[i]);
        file << "Task " << ids[i] << ": sin " << args[i] << " = " << res << "\n";
    }
}

void add_task_sqrt(Server<double>& server, int N, int client_id)
{
    std::ofstream file("client_sqrt.txt");
    std::vector<size_t> ids;
    std::vector<double> args;
    
    for (int i = 0; i < N; i++)
    {
        double arg = ((double)rand() / RAND_MAX) * 1000;
        std::function<double()> task = std::bind(func_sqrt<double>, arg);
        ids.push_back(server.add_task(task));
        args.push_back(arg);
    }
    
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < ids.size(); i++)
    {
        double res = server.request_result(ids[i]);
        file << "Task " << ids[i] << ": sqrt " << args[i] << " = " << res << "\n";
    }
}

void add_task_pow(Server<double>& server, int N, int client_id)
{
    std::ofstream file("client_pow.txt");
    std::vector<size_t> ids;
    std::vector<double> args;
    
    for (int i = 0; i < N; i++)
    {
        double arg = ((double)rand() / RAND_MAX) * 1000;
        std::function<double()> task = std::bind(func_pow<double>, arg, 2);
        ids.push_back(server.add_task(task));
        args.push_back(arg);
    }
    
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < ids.size(); i++)
    {
        double res = server.request_result(ids[i]);
        file << "Task " << ids[i] << ": pow " << args[i] << "^2 = " << res << "\n";
    }
}

bool test_results()
{
    std::ifstream sin_file("client_sin.txt");
    std::ifstream sqrt_file("client_sqrt.txt");
    std::ifstream pow_file("client_pow.txt");
    
    std::string line;
    int sin_count = 0, sqrt_count = 0, pow_count = 0;
    int sin_errors = 0, sqrt_errors = 0, pow_errors = 0;
    const double epsilon = 1e-5;
    
    std::regex sin_regex(R"(Task\s+(\d+):\s+sin\s+([\d.]+)\s+=\s+([\d.eE+-]+))");
    std::regex sqrt_regex(R"(Task\s+(\d+):\s+sqrt\s+([\d.]+)\s+=\s+([\d.eE+-]+))");
    std::regex pow_regex(R"(Task\s+(\d+):\s+pow\s+([\d.]+)\^(\d+)\s+=\s+([\d.eE+-]+))");
    
    std::smatch match;
    
    while (std::getline(sin_file, line))
    {
        sin_count++;
        
        if (std::regex_match(line, match, sin_regex))
        {
            double arg = std::stod(match[2]);
            double result = std::stod(match[3]);
            double expected = std::sin(arg);
            
            if (std::abs(result - expected) > epsilon) { sin_errors++; }
        }
    }

    while (std::getline(sqrt_file, line))
    {
        sqrt_count++;
        
        if (std::regex_match(line, match, sqrt_regex))
        {
            double arg = std::stod(match[2]);
            double result = std::stod(match[3]);
            double expected = std::sqrt(arg);
            
            if (std::abs(result - expected) > epsilon) { sqrt_errors++; }
        }
    }

    while (std::getline(pow_file, line))
    {
        pow_count++;
        
        if (std::regex_match(line, match, pow_regex))
        {
            double base = std::stod(match[2]);
            double exponent = std::stod(match[3]);
            double result = std::stod(match[4]);
            double expected = std::pow(base, exponent);
            
            if (std::abs(result - expected) > epsilon) { pow_errors++; }
        }
    }
    
    bool all_correct = (sin_errors == 0 && sqrt_errors == 0 && pow_errors == 0);
    return all_correct;
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));
    const int N = 10000;

    Server<double> server;
    std::cout << "start\n";
    server.start(16);
    
    std::thread client1(add_task_sin, std::ref(server), N, 1);
    std::thread client2(add_task_sqrt, std::ref(server), N, 2);
    std::thread client3(add_task_pow, std::ref(server), N, 3);
    
    client1.join();
    client2.join();
    client3.join();
    server.stop();

    bool result = test_results();
    std::cout << result << "\n";
    return result ? 0 : 1;
}