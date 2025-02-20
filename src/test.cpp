#include <blust/blust.hpp>
#include <iostream>
#include <random>
#include <chrono>

using namespace blust;
static size_t n_matrices = 2048;
constexpr int m_size     = 512;

void run_test(std::unique_ptr<matrix<int>[]>& m, std::unique_ptr<std::vector<int>[]>& v)
{
    for (size_t i = 0; i < n_matrices; i++)
        auto r = m[i] * v[i];
}

int main(int argc, char** argv)
{
    if (argc == 2)
        n_matrices = std::stoi(argv[1]);

    using namespace std::chrono;

    auto setup_start = high_resolution_clock::now();

    std::unique_ptr<matrix<int>[]> m;
    std::unique_ptr<std::vector<int>[]> v;

    std::uniform_int_distribution<size_t> dist(-8, 8);
    std::mt19937 rd(0x144258);

    m.reset(new matrix<int>[n_matrices]);
    v.reset(new std::vector<int>[n_matrices]);

    for (size_t i = 0; i < n_matrices; i++)
    {
        m[i].build({m_size, m_size});
        v[i].resize(m_size);

        const size_t size = m[i].size();
        for (size_t j = 0; j < size; j++)
            m[i](j) = dist(rd);
        
        for(size_t j = 0; j < v[i].size(); j++)
            v[i][j] = dist(rd);
    }


    auto start = high_resolution_clock::now();
    run_test(m, v);

    std::cout 
    << "N: " << n_matrices 
    << " setup time: " << duration_cast<milliseconds>(high_resolution_clock::now() - setup_start).count() << "ms "
    <<  " exec time: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << "ms\n";

    return 0;
}