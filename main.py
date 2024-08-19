import random
import matplotlib.pyplot as plt

class Chat:
    def __init__(self):
        self.cache_tokens = 0
        self.token_cnt = 0
        self.cost = 0

    def infer(self, tokens=1):
        self.token_cnt += tokens
        non_cache_tokens = self.token_cnt - self.cache_tokens
        self.cost += (3 * non_cache_tokens + 0.3 * self.cache_tokens) / 1e6

    def make_cache(self):
        self.cache_tokens = self.token_cnt
        self.cost += 3.75 * self.cache_tokens / 1e6

class Tracker(Chat):
    def __init__(self, pi=0.72):
        super().__init__()
        self.tracker = 0
        self.pi = pi

    def infer(self, tokens=1):
        ratio = self.cache_tokens / self.tracker if self.tracker else 0
        if ratio < self.pi:
            self.make_cache()
            self.tracker = 0

        super().infer(tokens)
        self.tracker += (self.token_cnt - self.cache_tokens)


def simulate_cost(pi, num_iterations):
    tracker = Tracker(pi)
    baseline = Chat()

    cost_ratios = []
    for _ in range(num_iterations):
        token_len = random.randint(100, 500)
        tracker.infer(token_len)
        baseline.infer(token_len)
        cost_ratios.append(baseline.cost / tracker.cost)

    print(f'Tracker cost: ${tracker.cost:.3f}, Baseline cost: ${baseline.cost:.3f}, '
          f'Cost ratio: {baseline.cost / tracker.cost:.3f}')

    return cost_ratios

def plot_cost_ratios(pi_values, num_iterations):
    plt.figure(figsize=(12, 6))

    for pi in pi_values:
        cost_ratios = simulate_cost(pi, num_iterations)
        plt.plot(cost_ratios, label=f'π={pi}')

    plt.xlabel('Iteration')
    plt.ylabel('Cost Ratio (Baseline / Tracker)')
    plt.title('Cost Ratio Over Time for Different π Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_iterations = 80
    pi_values = [0.5, 0.72, 1.5, 2, 0.5]
    plot_cost_ratios(pi_values, num_iterations)
