#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace achmm {

constexpr int ALPHABET_SIZE = 4;
constexpr int INVALID_SYMBOL = -1;

inline int encode_base(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return INVALID_SYMBOL;
    }
}

struct ACHMMParams {
    int K = 8;
    int D = 3;
    double dirichlet_alpha = 10.0;
    double occupancy_tau = 5.0;
    double convergence_delta = 1e-6;
    int max_iterations = 200;
};

struct TrainResult {
    double log_likelihood = 0.0;
    int iterations = 0;
    bool converged = false;
    std::uint64_t flops = 0;
};

class ACHMMTrellis {
public:
    explicit ACHMMTrellis(const ACHMMParams& params);

  void set_random_seed(std::uint64_t seed);

  TrainResult fit(
      const std::vector<int>& symbols,
      const std::vector<uint8_t>& active_mask,
      const std::vector<double>& init_pi,
      const std::vector<double>& init_A,
      const std::vector<double>& init_B);

  double score(
      const std::vector<int>& symbols,
      const std::vector<uint8_t>& active_mask);

  const std::vector<double>& pi() const { return pi_; }
  const std::vector<double>& A() const { return A_; }
  const std::vector<double>& B() const { return B_; }

  int num_states() const { return params_.K; }
  int context_depth() const { return params_.D; }
  int num_contexts() const { return num_contexts_; }

private:
    ACHMMParams params_;
    int num_contexts_;
    std::uint64_t rng_state_;

    std::vector<double> pi_;
    std::vector<double> A_;
    std::vector<double> B_;
    std::vector<int> context_indices_;

    double rand_uniform();
    void init_uniform_params();
    void precompute_contexts(const std::vector<int>& symbols);
    int context_index_at(int t) const;

    double emission_prob(int k, int t, int symbol) const;
    bool is_active(int t, const std::vector<uint8_t>& mask) const;

    double forward_backward(
        const std::vector<int>& symbols,
        const std::vector<uint8_t>& active_mask,
        std::vector<double>& gamma,
        std::vector<double>& xi,
        std::uint64_t& flops) const;

    void m_step(
        const std::vector<int>& symbols,
        const std::vector<uint8_t>& active_mask,
        const std::vector<double>& gamma,
        const std::vector<double>& xi);
};

}  // namespace achmm
