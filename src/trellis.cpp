#include "trellis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace achmm {

namespace {

constexpr double LOG_ZERO = -1e300;
constexpr double MIN_PROB = 1e-12;

inline double safe_log(double x) {
    return std::log(std::max(x, MIN_PROB));
}

}  // namespace

ACHMMTrellis::ACHMMTrellis(const ACHMMParams& params)
    : params_(params),
      num_contexts_(static_cast<int>(std::pow(ALPHABET_SIZE, params.D))),
      rng_state_(42ULL) {
    if (params_.K < 1) throw std::invalid_argument("K must be >= 1");
    if (params_.D < 0) throw std::invalid_argument("D must be >= 0");
    init_uniform_params();
}

void ACHMMTrellis::set_random_seed(std::uint64_t seed) { rng_state_ = seed; }

double ACHMMTrellis::rand_uniform() {
    rng_state_ = rng_state_ * 6364136223846793005ULL + 1ULL;
    return static_cast<double>((rng_state_ >> 11) & 0xFFFFFFFFULL) / 4294967296.0;
}

void ACHMMTrellis::init_uniform_params() {
    const int K = params_.K;
    const int H = std::max(1, num_contexts_);
    pi_.assign(K, 1.0 / K);
    A_.assign(K * K, 1.0 / K);
    B_.assign(K * H * ALPHABET_SIZE, 1.0 / ALPHABET_SIZE);
}

void ACHMMTrellis::precompute_contexts(const std::vector<int>& symbols) {
    const int N = static_cast<int>(symbols.size());
    const int D = params_.D;
    context_indices_.assign(N, 0);
    if (D == 0) return;

    const int base = ALPHABET_SIZE;
    for (int t = 0; t < N; ++t) {
        if (t < D) {
            context_indices_[t] = 0;
            continue;
        }
        int idx = 0;
        bool valid = true;
        for (int d = 0; d < D; ++d) {
            const int sym = symbols[t - D + d];
            if (sym < 0 || sym >= ALPHABET_SIZE) {
                valid = false;
                break;
            }
            idx = idx * base + sym;
        }
        context_indices_[t] = valid ? idx : 0;
    }
}

int ACHMMTrellis::context_index_at(int t) const {
    if (t < 0 || t >= static_cast<int>(context_indices_.size())) return 0;
    return context_indices_[t];
}

bool ACHMMTrellis::is_active(int t, const std::vector<uint8_t>& mask) const {
    if (mask.empty()) return true;
    return mask[t] != 0;
}

double ACHMMTrellis::emission_prob(int k, int t, int symbol) const {
    if (symbol < 0 || symbol >= ALPHABET_SIZE) return 1.0;
    const int h = context_index_at(t);
    const int H = std::max(1, num_contexts_);
    const std::size_t idx =
        (static_cast<std::size_t>(k) * H + h) * ALPHABET_SIZE + symbol;
    return B_[idx];
}

double ACHMMTrellis::forward_backward(
    const std::vector<int>& symbols,
    const std::vector<uint8_t>& active_mask,
    std::vector<double>& gamma,
    std::vector<double>& xi,
    std::uint64_t& flops) const {
    const int N = static_cast<int>(symbols.size());
    const int K = params_.K;
    if (N == 0) return 0.0;

    std::vector<double> log_alpha(N * K, LOG_ZERO);
    std::vector<double> log_beta(N * K, LOG_ZERO);

    auto la = [&](int t, int k) -> double& { return log_alpha[t * K + k]; };
    auto lb = [&](int t, int k) -> double& { return log_beta[t * K + k]; };

    for (int k = 0; k < K; ++k) {
        double emit = emission_prob(k, 0, symbols[0]);
        if (!is_active(0, active_mask)) emit = 1.0;
        la(0, k) = safe_log(pi_[k]) + safe_log(emit);
    }
    flops += static_cast<std::uint64_t>(K);

    for (int t = 1; t < N; ++t) {
        for (int j = 0; j < K; ++j) {
            double sum = LOG_ZERO;
            for (int i = 0; i < K; ++i) {
                const double v = la(t - 1, i) + safe_log(A_[i * K + j]);
                sum = std::log(std::exp(sum - v) + 1.0) + v;
            }
            double emit = emission_prob(j, t, symbols[t]);
            if (!is_active(t, active_mask)) emit = 1.0;
            la(t, j) = sum + safe_log(emit);
        }
        flops += static_cast<std::uint64_t>(K * K);
    }

    double log_likelihood = LOG_ZERO;
    for (int k = 0; k < K; ++k) {
        const double v = la(N - 1, k);
        log_likelihood = std::log(std::exp(log_likelihood - v) + 1.0) + v;
    }

    for (int k = 0; k < K; ++k) lb(N - 1, k) = 0.0;

    for (int t = N - 2; t >= 0; --t) {
        for (int i = 0; i < K; ++i) {
            double sum = LOG_ZERO;
            for (int j = 0; j < K; ++j) {
                double emit = emission_prob(j, t + 1, symbols[t + 1]);
                if (!is_active(t + 1, active_mask)) emit = 1.0;
                const double v = safe_log(A_[i * K + j]) + safe_log(emit) + lb(t + 1, j);
                sum = std::log(std::exp(sum - v) + 1.0) + v;
            }
            lb(t, i) = sum;
        }
        flops += static_cast<std::uint64_t>(K * K);
    }

    gamma.assign(N * K, 0.0);
    xi.assign((N - 1) * K * K, 0.0);

    for (int t = 0; t < N; ++t) {
        double denom = LOG_ZERO;
        for (int k = 0; k < K; ++k) {
            const double v = la(t, k) + lb(t, k);
            denom = std::log(std::exp(denom - v) + 1.0) + v;
        }
        for (int k = 0; k < K; ++k) {
            gamma[t * K + k] = std::exp(la(t, k) + lb(t, k) - denom);
        }
    }

    for (int t = 0; t < N - 1; ++t) {
        double denom = LOG_ZERO;
        std::vector<double> numer(K * K, 0.0);
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                double emit = emission_prob(j, t + 1, symbols[t + 1]);
                if (!is_active(t + 1, active_mask)) emit = 1.0;
                const double v = la(t, i) + safe_log(A_[i * K + j]) + safe_log(emit) + lb(t + 1, j);
                numer[i * K + j] = std::exp(v);
                denom = std::log(std::exp(denom - v) + 1.0) + v;
            }
        }
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                xi[t * K * K + i * K + j] = numer[i * K + j] * std::exp(-denom);
            }
        }
        flops += static_cast<std::uint64_t>(K * K);
    }

    return log_likelihood;
}

void ACHMMTrellis::m_step(
    const std::vector<int>& symbols,
    const std::vector<uint8_t>& active_mask,
    const std::vector<double>& gamma,
    const std::vector<double>& xi) {
    const int N = static_cast<int>(symbols.size());
    const int K = params_.K;
    const int H = std::max(1, num_contexts_);
    const double alpha = params_.dirichlet_alpha;
    const double tau = params_.occupancy_tau;

    std::vector<double> bg(K * ALPHABET_SIZE, 0.0);
    std::vector<double> bg_denom(K, 0.0);
    for (int t = 0; t < N; ++t) {
        if (!is_active(t, active_mask)) continue;
        const int a = symbols[t];
        if (a < 0 || a >= ALPHABET_SIZE) continue;
        for (int k = 0; k < K; ++k) {
            bg[k * ALPHABET_SIZE + a] += gamma[t * K + k];
            bg_denom[k] += gamma[t * K + k];
        }
    }
    for (int k = 0; k < K; ++k) {
        const double d = bg_denom[k] + alpha * ALPHABET_SIZE;
        for (int a = 0; a < ALPHABET_SIZE; ++a) {
            bg[k * ALPHABET_SIZE + a] = (bg[k * ALPHABET_SIZE + a] + alpha) / d;
        }
    }

    for (int i = 0; i < K; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < K; ++j) {
            double num = 0.0;
            for (int t = 0; t < N - 1; ++t) num += xi[t * K * K + i * K + j];
            A_[i * K + j] = num;
            row_sum += num;
        }
        if (row_sum < MIN_PROB) {
            for (int j = 0; j < K; ++j) A_[i * K + j] = 1.0 / K;
        } else {
            for (int j = 0; j < K; ++j) A_[i * K + j] /= row_sum;
        }
    }

    std::vector<double> emit_num(K * H * ALPHABET_SIZE, 0.0);
    std::vector<double> emit_den(K * H, 0.0);
    std::vector<double> occ(K * H, 0.0);

    for (int t = 0; t < N; ++t) {
        if (!is_active(t, active_mask)) continue;
        const int a = symbols[t];
        if (a < 0 || a >= ALPHABET_SIZE) continue;
        const int h = context_index_at(t);
        for (int k = 0; k < K; ++k) {
            const std::size_t cell = static_cast<std::size_t>(k) * H + h;
            emit_num[cell * ALPHABET_SIZE + a] += gamma[t * K + k];
            emit_den[cell] += gamma[t * K + k];
            occ[cell] += gamma[t * K + k];
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int h = 0; h < H; ++h) {
            const std::size_t cell = static_cast<std::size_t>(k) * H + h;
            const double weight = std::min(1.0, occ[cell] / tau);
            const double denom = emit_den[cell] + alpha * ALPHABET_SIZE;
            for (int a = 0; a < ALPHABET_SIZE; ++a) {
                double smoothed = (emit_num[cell * ALPHABET_SIZE + a] + alpha) / denom;
                double background = bg[k * ALPHABET_SIZE + a];
                B_[cell * ALPHABET_SIZE + a] = weight * smoothed + (1.0 - weight) * background;
            }
            double row_sum = 0.0;
            for (int a = 0; a < ALPHABET_SIZE; ++a) row_sum += B_[cell * ALPHABET_SIZE + a];
            for (int a = 0; a < ALPHABET_SIZE; ++a) B_[cell * ALPHABET_SIZE + a] /= row_sum;
        }
    }

    double pi_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        pi_[k] = gamma[k];
        pi_sum += pi_[k];
    }
    for (int k = 0; k < K; ++k) pi_[k] /= pi_sum;
}

TrainResult ACHMMTrellis::fit(
    const std::vector<int>& symbols,
    const std::vector<uint8_t>& active_mask,
    const std::vector<double>& init_pi,
    const std::vector<double>& init_A,
    const std::vector<double>& init_B) {
    if (!init_pi.empty()) pi_ = init_pi;
    if (!init_A.empty()) A_ = init_A;
    if (!init_B.empty()) B_ = init_B;

    precompute_contexts(symbols);

    TrainResult result;
    double prev_ll = std::numeric_limits<double>::lowest();
    std::vector<double> gamma;
    std::vector<double> xi;

    for (int it = 0; it < params_.max_iterations; ++it) {
        std::uint64_t iter_flops = 0;
        const double ll = forward_backward(symbols, active_mask, gamma, xi, iter_flops);
        result.flops += iter_flops;
        m_step(symbols, active_mask, gamma, xi);

        if (it > 0 && std::abs(ll - prev_ll) < params_.convergence_delta) {
            result.converged = true;
            result.iterations = it + 1;
            result.log_likelihood = ll;
            break;
        }
        prev_ll = ll;
        result.log_likelihood = ll;
        result.iterations = it + 1;
    }

    return result;
}

double ACHMMTrellis::score(
    const std::vector<int>& symbols,
    const std::vector<uint8_t>& active_mask) {
    precompute_contexts(symbols);

    std::vector<double> gamma;
    std::vector<double> xi;
    std::uint64_t flops = 0;
    return forward_backward(symbols, active_mask, gamma, xi, flops);
}

}  // namespace achmm
