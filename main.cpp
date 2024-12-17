#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <parlay/primitives.h>
#include <parlay/parallel.h>

struct graph {
    size_t n;
    std::vector<std::vector<size_t>> neighbours;

    graph(size_t n): n(n), neighbours(n) {}

    void add_edge(size_t u, size_t v, bool bidirectional=false) {
        neighbours[u].push_back(v);
        if (bidirectional) neighbours[v].push_back(u);
    }
};

namespace Seq {
    std::vector<std::vector<size_t>> bfs(size_t s, const graph& g) {
        std::vector<std::vector<size_t>> layers;
        std::vector<char> visited(g.n);
        layers.push_back({s});
        visited[s] = true;
        while (!layers.back().empty()) {
            layers.emplace_back();
            const auto& cur_layer = layers[layers.size() - 2];
            auto& next_layer = layers.back();
            for (size_t u: cur_layer) {
                for (size_t v: g.neighbours[u]) {
                    if (visited[v]) continue;
                    visited[v] = true;
                    next_layer.push_back(v);
                }
            }
        }
        return layers;
    }
} // namespace Seq

namespace Par {
    struct options {
        bool use_library_flatten = true;
        bool return_parlay_sequence = true;
    };

    template<options opts>
    auto get_all_neighbours(const auto& vertices, const graph& g) {
        if constexpr (opts.use_library_flatten) {
            return parlay::flatten(
                parlay::delayed_map(vertices, [&g](size_t u) -> const std::vector<size_t>& {
                    return g.neighbours[u]; 
                })
            );
        } else {
            auto offsets = parlay::map(vertices, [&g](size_t u) { return g.neighbours[u].size(); });
            size_t sz = parlay::scan_inplace(offsets);
            auto all_neighbours = parlay::sequence<size_t>::uninitialized(sz);
            parlay::parallel_for(0, vertices.size(), [&](size_t i) {
                size_t u = vertices[i];
                auto it1 = all_neighbours.begin() + offsets[i];
                auto it2 = g.neighbours[u].begin(); 
                parlay::parallel_for(0, g.neighbours[u].size(), [&](size_t j) {
                    parlay::assign_uninitialized(*(it1 + j), *(it2 + j));
                }, 1000);
            });
            return all_neighbours;
        }
    }

    template<options opts = options{}>
    auto bfs(size_t s, const graph& g) {
        parlay::sequence<parlay::sequence<size_t>> layers;
        parlay::sequence<std::atomic_bool> visited(g.n);
        layers.push_back({s});
        visited[s] = true;
        while (!layers.back().empty()) {
            const auto& cur_layer = layers.back();
            auto all_neighbours = get_all_neighbours<opts>(cur_layer, g);
            layers.push_back(
                parlay::filter(all_neighbours, [&visited](size_t u) {
                    bool val = false;
                    return !visited[u] && visited[u].compare_exchange_strong(val, true);
                })
            );
        }
        if constexpr (opts.return_parlay_sequence) {
            return layers;
        } else {
            return parlay::map(layers, [](const auto& x) { return x.to_vector();}).to_vector();
        }
    }
} // namespace Par

namespace testing {
    constexpr size_t CUBE_SIZE = 400;

    class random_generator {
        std::mt19937 eng{42};
    public:
        template <typename T>
        std::vector<T> rand_vec(size_t size) {
            std::uniform_int_distribution<T> gen;
            std::vector<T> arr(size);
            for (int j = 0; j < arr.size(); ++j) {
                arr[j] = gen(eng);
            }
            return arr;
        }

        template <typename T=size_t>
        T rand(T low, T high) {
            return std::uniform_int_distribution<T>{low, high}(eng);
        }

        template <typename T=size_t>
        T rand(T high=std::numeric_limits<T>::max()) {
            return std::uniform_int_distribution<T>{0, high}(eng);
        }
    } gen;

    template <typename T>
    struct Func {
        std::string name;
        T func;
    };

    graph get_test_graph() {
        graph g(CUBE_SIZE * CUBE_SIZE * CUBE_SIZE);
        auto get_vertex_id = [](int i, int j, int k) {
            return i * CUBE_SIZE * CUBE_SIZE + j * CUBE_SIZE + k;
        };
        auto valid_vertex = [](int i, int j, int k) {
            return 0 <= i && i < CUBE_SIZE
                && 0 <= j && j < CUBE_SIZE
                && 0 <= k && k < CUBE_SIZE;
        };
        auto neighbours = [](int i, int j, int k) {
            return std::array{
                std::tuple{i + 1, j, k},
                std::tuple{i - 1, j, k},
                std::tuple{i, j + 1, k},
                std::tuple{i, j - 1, k},
                std::tuple{i, j, k + 1},
                std::tuple{i, j, k - 1}
            };
        };
        for (int i = 0; i < CUBE_SIZE; ++i) {
            for (int j = 0; j < CUBE_SIZE; ++j) {
                for (int k = 0; k < CUBE_SIZE; ++k) {
                    int u = get_vertex_id(i, j, k);
                    for (auto [i1, j1, k1]: neighbours(i, j, k)) {
                        if (valid_vertex(i1, j1, k1))
                            g.add_edge(u, get_vertex_id(i1, j1, k1));
                    }
                }
            }
        }
        return g;
    }

    template <typename... Funcs>
    auto test_speed(Funcs&&... funcs) {
        const int n_attempts = 5;
        using duration_t = std::chrono::duration<double>;
        std::map<std::string, duration_t> total;
        const graph g = get_test_graph();
        const size_t start = 0;

        std::cout << "\033[1mBenchmark:\033[0m\n";
        for (int i = 1; i <= n_attempts; ++i) {
            std::cout << "\033[1mTest " << i << ":\033[0m\n";
            ([&] {
                auto t1 = std::chrono::high_resolution_clock::now();
                funcs.func(start, g);
                auto t2 = std::chrono::high_resolution_clock::now();
                duration_t d = t2 - t1;
                total[funcs.name] += d;
                std::cout << funcs.name << ": " << d.count() << '\n';
            } (), ...);
        }
        std::cout << "\033[1mTotal:\033[0m\n";
        ((std::cout << funcs.name << ": " << total[funcs.name].count() << '\n'), ...);
        return total;
    }

    template <typename F>
    bool test_graph(size_t s, const graph& g, F func) {
        auto res = func(s, g);
        auto ans = Seq::bfs(s, g);
        
        if (res.size() != ans.size()) return false;
        for (int l = 0; l < res.size(); ++l) {
            std::sort(res[l].begin(), res[l].end());
            std::sort(ans[l].begin(), ans[l].end());
            if (res[l] != ans[l]) return false;
        }
        return true;
    }

    std::string result(bool res) {
        return res ? "\033[0;32mPassed\033[0m" : "\033[0;31mFailed\033[0m";
    }

    template <typename F>
    void test_correctness(F func) {
        std::mt19937 engine(123);
        using arr_t = size_t;
        random_generator gen;
        std::cout << "\033[1mTest correctness:\033[0m\n";
        {
            graph g(1);
            std::cout << "One vertex: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(1);
            g.add_edge(0, 0);
            std::cout << "One vertex and loop: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(1);
            g.add_edge(0, 0);
            g.add_edge(0, 0);
            g.add_edge(0, 0);
            std::cout << "One vertex and many loops: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(2);
            g.add_edge(0, 1, true);
            std::cout << "Single edge: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(2);
            g.add_edge(0, 1, true);
            g.add_edge(0, 1, true);
            g.add_edge(0, 1, true);
            std::cout << "Multi-edge: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(3);
            std::cout << "Isolated vertices: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            graph g(3);
            g.add_edge(0, 1, true);
            g.add_edge(0, 2, true);
            g.add_edge(1, 2, true);
            std::cout << "Small cycle: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            const int n = 8;
            graph g(n);
            for (int i = 0; i < n; ++i) {
                g.add_edge(i, (i + 1) % n, true);
            }
            std::cout << "Big cycle: " << result(test_graph(0, g, func)) << '\n';
        }
        {
            const int n = 500;
            graph g(n);
            for (int i = 1; i < n; ++i) {
                g.add_edge(i, gen.rand(i - 1), true);
            }
            std::cout << "Random tree, start from root: " << result(test_graph(0, g, func)) << '\n';
            {
                bool res = true;
                for (int i = 1; i < n; ++i) {
                    res &= test_graph(i, g, func);
                }
                std::cout << "Random tree, random start: " << result(res) << '\n';
            }
        }
        {
            bool res = true;
            for (int i = 0; i < 10 && res; ++i) {
                size_t sz = gen.rand(100, 2000);
                for (int j = 0; j < 100 && res; ++j) {
                    graph g(sz);
                    size_t m = gen.rand(sz * sz);
                    for (int i = 0; i < m; ++i) {
                        g.add_edge(gen.rand(sz - 1), gen.rand(sz - 1));
                    }

                    res &= test_graph(0, g, func);
                }
            }
            std::cout << "Random tests: " << result(res) << '\n';
        }
        std::cout << '\n';
    }
} // namespace testing


int main() {
    std::cout << "Number of available threads: " << parlay::num_workers() << "\n\n";

    testing::test_correctness(Par::bfs<Par::options{true, false}>);
    testing::test_correctness(Par::bfs<Par::options{false, false}>);

    auto res = testing::test_speed(
        testing::Func("seq", Seq::bfs),
        testing::Func("par", Par::bfs<Par::options{true, false}>),
        testing::Func("par2", Par::bfs<Par::options{false, false}>),
        testing::Func("par3", Par::bfs<Par::options{true, true}>)
    );
    std::cout << "Speedup: " << res["seq"] / res["par"] << "x\n";
}
