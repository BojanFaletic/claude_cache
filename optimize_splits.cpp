#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>
#include <stack>
#include <cmath>
#include <omp.h>

double sim(const std::vector<int>& times, int L) {
    int cache_tokens = 0;
    int token_cnt = 0;
    double cost = 0.0;

    for (int n = 0; n < L; ++n) {
        ++token_cnt;
        int non_cache_tokens = token_cnt - cache_tokens;
        cost += (3 * non_cache_tokens + 0.3 * cache_tokens) / 1e6;

        if (std::binary_search(times.begin(), times.end(), n)) {
            cache_tokens = token_cnt;
            cost += 3.75 * cache_tokens / 1e6;
        }
    }

    return cost;
}

double estimate_min_cost(const std::vector<int>& current_splits, int depth, int L, int N) {
    if (depth == 0) return 0.0;
    int last_split = current_splits[depth - 1];
    int remaining_splits = N - depth;
    int remaining_length = L - last_split;
    return (remaining_splits * remaining_length) * (3 + 0.3 * depth) / 1e6;
}

struct SearchState {
    int depth;
    int last_split;
    std::vector<int> current_splits;
};

std::pair<double, std::vector<int>> optimize_splits(int N, int L) {
    double global_best_cost = std::numeric_limits<double>::infinity();
    std::vector<int> global_best_splits(N);

    #pragma omp parallel
    {
        double thread_best_cost = std::numeric_limits<double>::infinity();
        std::vector<int> thread_best_splits(N);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i <= L - N; ++i) {
            std::vector<int> current_splits(N, 0);
            current_splits[0] = i;

            std::stack<SearchState> stack;
            stack.push({1, i, current_splits});

            while (!stack.empty()) {
                auto [depth, last_split, current_splits] = stack.top();
                stack.pop();

                if (depth == N) {
                    double cost = sim(current_splits, L);
                    if (cost < thread_best_cost) {
                        thread_best_cost = cost;
                        thread_best_splits = current_splits;
                    }
                    continue;
                }

                for (int j = last_split + 1; j <= L - (N - depth); ++j) {
                    current_splits[depth] = j;
                    double estimated_min_cost = estimate_min_cost(current_splits, depth + 1, L, N);
                    if (estimated_min_cost < thread_best_cost) {
                        stack.push({depth + 1, j, current_splits});
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (thread_best_cost < global_best_cost) {
                global_best_cost = thread_best_cost;
                global_best_splits = thread_best_splits;
            }
        }
    }

    return {global_best_cost, global_best_splits};
}

std::tuple<int, double, std::vector<int>> find_optimal_N(int max_N, int L) {
    double optimal_cost = std::numeric_limits<double>::infinity();
    std::vector<int> optimal_splits;
    int optimal_N = 0;

    for (int N = 1; N <= max_N; ++N) {
        auto [cost, splits] = optimize_splits(N, L);

        std::cout << "N = " << N << ", Cost: " << cost << ", Splits: ";
        for (int split : splits) std::cout << split << " ";
        std::cout << std::endl;

        if (cost < optimal_cost) {
            optimal_cost = cost;
            optimal_splits = splits;
            optimal_N = N;
        }
    }

    return {optimal_N, optimal_cost, optimal_splits};
}

int main() {
    int L = 100;  // Max context length
    int max_N = 7;  // Maximum number of splits to consider

    auto start_time = std::chrono::high_resolution_clock::now();

    auto [optimal_N, optimal_cost, optimal_splits] = find_optimal_N(max_N, L);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Optimal N: " << optimal_N << ", Cost: " << optimal_cost << ", Splits: ";
    for (int split : optimal_splits) std::cout << split << " ";
    std::cout << std::endl;

    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;

    // calculate cost with 3USD per input as reference
    std::cout << "Cost per input: " << 3 * std::pow(L, 2) / 2e6 << "USD" << std::endl;

    return 0;
}