// llama-perplexity --kl-divergence-base format and statistics. Host only.
// Keep the float intermediates and percentile arithmetic aligned with
// tools/perplexity/perplexity.cpp in llama.cpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace b70 {

class KldBaseReader {
public:
    explicit KldBaseReader(const std::string& path) : path_(path), in_(path, std::ios::binary | std::ios::ate) {
        if (!in_) throw std::runtime_error("--ppl-kld-base cannot open " + path);
        const auto end = in_.tellg();
        if (end < 20) throw std::runtime_error("--ppl-kld-base truncated header: " + path);
        in_.seekg(0);
        char magic[8];
        read(magic, sizeof(magic));
        if (std::memcmp(magic, "_logits_", 8)) throw std::runtime_error("--ppl-kld-base invalid magic: " + path);
        read(&ctx_, sizeof(ctx_));
        read(&vocab_, sizeof(vocab_));
        read(&chunks_, sizeof(chunks_));
        if (ctx_ < 4 || vocab_ <= 0 || chunks_ <= 0)
            throw std::runtime_error("--ppl-kld-base invalid n_ctx, n_vocab, or n_chunk: " + path);
        scored_ = ctx_ - 1 - ctx_/2;
        // Each row starts with two float32 values (four uint16), then n_vocab
        // quantized log probabilities, padded to an even number of uint16s.
        row_words_ = 2*((uint64_t(vocab_) + 1)/2) + 4;
        token_bytes_ = checked(checked(uint64_t(ctx_), uint64_t(chunks_)), 4);
        row_bytes_ = checked(checked(uint64_t(scored_), row_words_), 2);
        const uint64_t records = checked(row_bytes_, uint64_t(chunks_));
        if (token_bytes_ > UINT64_MAX - 20 || records > UINT64_MAX - 20 - token_bytes_)
            throw std::runtime_error("--ppl-kld-base size overflow: " + path);
        records_offset_ = 20 + token_bytes_;
        const uint64_t expected = records_offset_ + records;
        if (uint64_t(end) != expected)
            throw std::runtime_error("--ppl-kld-base size mismatch (truncated or trailing data): " + path);
    }

    int ctx() const { return ctx_; }
    int vocab() const { return vocab_; }
    int chunks() const { return chunks_; }
    int scored_per_chunk() const { return scored_; }
    size_t row_words() const { return size_t(row_words_); }

    void read_chunk(int chunk, std::vector<int32_t>& tokens, std::vector<uint16_t>& log_probs) {
        if (chunk < 0 || chunk >= chunks_) throw std::out_of_range("--ppl-kld-base chunk index");
        tokens.resize(size_t(ctx_));
        log_probs.resize(size_t(row_words_ * uint64_t(scored_)));
        in_.seekg(std::streamoff(20 + uint64_t(chunk)*uint64_t(ctx_)*4));
        read(tokens.data(), uint64_t(ctx_)*4);
        in_.seekg(std::streamoff(records_offset_ + uint64_t(chunk)*row_bytes_));
        read(log_probs.data(), row_bytes_);
    }

private:
    static uint64_t checked(uint64_t a, uint64_t b) {
        if (b && a > UINT64_MAX/b) throw std::runtime_error("--ppl-kld-base size overflow");
        return a*b;
    }
    void read(void* dst, uint64_t bytes) {
        if (bytes > uint64_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("--ppl-kld-base read too large: " + path_);
        if (!in_.read(static_cast<char*>(dst), std::streamsize(bytes)))
            throw std::runtime_error("--ppl-kld-base truncated while reading: " + path_);
    }
    std::string path_;
    std::ifstream in_;
    int32_t ctx_ = 0, vocab_ = 0, chunks_ = 0, scored_ = 0;
    uint64_t row_words_ = 0, token_bytes_ = 0, row_bytes_ = 0, records_offset_ = 0;
};

struct KldStats {
    double sum_nll = 0, sum_nll2 = 0, sum_nll_base = 0, sum_nll_base2 = 0;
    double sum_nll_nll_base = 0, sum_kld = 0, sum_kld2 = 0;
    double sum_p_diff = 0, sum_p_diff2 = 0, sum_p_diff4 = 0;
    float max_p_diff = 0;
    size_t n_same_top = 0, count = 0;
    std::vector<float> kld_values, p_diff_values;

    // Returns KLD and token probability difference. The stored base row
    // includes the two float32 header values in its first four uint16s.
    std::pair<double, float> add(int n_vocab, const float* logits, const uint16_t* base_row, int tok) {
        if (n_vocab <= 0 || tok < 0 || tok >= n_vocab)
            throw std::runtime_error("KLD target outside vocabulary");
        float max_logit = logits[0];
        int imax = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > max_logit) { max_logit = logits[i]; imax = i; }
        }
        double sum_exp = 0;
        for (int i = 0; i < n_vocab; ++i) sum_exp += ::expf(logits[i] - max_logit);
        const float log_sum_exp = ::log(sum_exp);
        float scale, min_log_prob;
        std::memcpy(&scale, base_row, sizeof(scale));
        std::memcpy(&min_log_prob, base_row + 2, sizeof(min_log_prob));
        const uint16_t* q = base_row + 4;

        const float nll = max_logit + log_sum_exp - logits[tok];
        sum_nll += nll;
        sum_nll2 += nll*nll;
        const float nll_base = -(scale*q[tok] + min_log_prob);
        sum_nll_base += nll_base;
        sum_nll_base2 += nll_base*nll_base;
        sum_nll_nll_base += nll*nll_base;

        max_logit += log_sum_exp;
        double sum = 0;
        int imax_base = -1;
        float p_log_base_max = 0;
        for (int i = 0; i < n_vocab; ++i) {
            const float p_log_base = scale*q[i] + min_log_prob;
            if (i == 0 || p_log_base > p_log_base_max) {
                p_log_base_max = p_log_base;
                imax_base = i;
            }
            if (p_log_base > -16.f) {
                const float p_base = ::expf(p_log_base);
                sum += p_base * (p_log_base - logits[i] + max_logit);
            }
        }
        sum_kld += sum;
        sum_kld2 += sum*sum;
        ++count;
        if (imax == imax_base) ++n_same_top;
        const float p_base = ::expf(-nll_base);
        const float p = ::expf(-nll);
        const float p_diff = p - p_base;
        sum_p_diff += p_diff;
        const double p_diff2 = p_diff*p_diff;
        sum_p_diff2 += p_diff2;
        sum_p_diff4 += p_diff2*p_diff2;
        max_p_diff = std::max(max_p_diff, std::fabs(p_diff));
        kld_values.push_back(float(sum));
        p_diff_values.push_back(p_diff);
        return {sum, p_diff};
    }
};

inline std::pair<double, double> kld_mean_uncertainty(double sum, double sum2, size_t count) {
    if (!count) return {0, 0};
    const double mean = sum/count;
    const double var = sum2/count - mean*mean;
    return {mean, var > 0 && count > 10 ? std::sqrt(var/(count - 1)) : 0.};
}

inline double kld_covariance(double suma, double sumb, double sumab, size_t count) {
    if (count < 10) return 0.;
    return (sumab/count - (suma/count)*(sumb/count))/(count - 1);
}

inline float kld_percentile(const std::vector<float>& sorted, float fraction) {
    if (fraction <= 0) return sorted.front();
    if (fraction >= 1) return sorted.back();
    float p = fraction*(sorted.size() - 1);
    const size_t ip = size_t(p);
    p -= ip;
    return (1 - p)*sorted[ip] + p*sorted[std::min(ip + 1, sorted.size() - 1)];
}

inline float kld_median(const std::vector<float>& sorted) {
    return sorted.size()%2 == 0 ? 0.5f*(sorted[sorted.size()/2] + sorted[sorted.size()/2-1])
                                : sorted[sorted.size()/2];
}

inline void print_kld_statistics(KldStats& s) {
    // llama.cpp suppresses the summary for fewer than 100 scored tokens.
    if (s.count < 100) return;
    std::sort(s.kld_values.begin(), s.kld_values.end());
    std::sort(s.p_diff_values.begin(), s.p_diff_values.end());
    auto log_ppl = kld_mean_uncertainty(s.sum_nll, s.sum_nll2, s.count);
    auto log_base = kld_mean_uncertainty(s.sum_nll_base, s.sum_nll_base2, s.count);
    const double ppl = std::exp(log_ppl.first), ppl_unc = ppl*log_ppl.second;
    const double base = std::exp(log_base.first), base_unc = base*log_base.second;
    const double cov = kld_covariance(s.sum_nll, s.sum_nll_base, s.sum_nll_nll_base, s.count);
    const double ratio_log = log_ppl.first - log_base.first;
    const double ratio_log_unc = std::sqrt(log_ppl.second*log_ppl.second + log_base.second*log_base.second - 2*cov);
    const double ratio = std::exp(ratio_log);
    std::printf("====== Perplexity statistics ======\n");
    std::printf("Mean PPL(Q)                   : %10.6lf ± %10.6lf\n", ppl, ppl_unc);
    std::printf("Mean PPL(base)                : %10.6lf ± %10.6lf\n", base, base_unc);
    std::printf("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100*cov/(log_ppl.second*log_base.second));
    std::printf("Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf\n", ratio_log, ratio_log_unc);
    std::printf("Mean PPL(Q)/PPL(base)         : %10.6lf ± %10.6lf\n", ratio, ratio*ratio_log_unc);
    std::printf("Mean PPL(Q)-PPL(base)         : %10.6lf ± %10.6lf\n", ppl-base,
                std::sqrt(ppl_unc*ppl_unc + base_unc*base_unc - 2*ppl*base*cov));
    std::printf("\n====== KL divergence statistics ======\n");
    auto kld = kld_mean_uncertainty(s.sum_kld, s.sum_kld2, s.count);
    std::printf("Mean    KLD: %10.6lf ± %10.6lf\n", kld.first, kld.second);
    std::printf("Maximum KLD: %10.6f\n", s.kld_values.back());
    for (float q : {0.999f, 0.990f, 0.950f, 0.900f})
        std::printf("%4.1f%%   KLD: %10.6f\n", 100*q, kld_percentile(s.kld_values, q));
    std::printf("Median  KLD: %10.6f\n", kld_median(s.kld_values));
    for (float q : {0.100f, 0.050f, 0.010f, 0.001f})
        std::printf("%4.1f%%   KLD: %10.6f\n", 100*q, kld_percentile(s.kld_values, q));
    std::printf("Minimum KLD: %10.6f\n", s.kld_values.front());
    std::printf("\n====== Token probability statistics ======\n");
    auto diff = kld_mean_uncertainty(s.sum_p_diff, s.sum_p_diff2, s.count);
    std::printf("Mean    Δp: %6.3lf ± %5.3lf %%\n", 100*diff.first, 100*diff.second);
    std::printf("Maximum Δp: %6.3lf%%\n", 100.*s.p_diff_values.back());
    for (float q : {0.999f, 0.990f, 0.950f, 0.900f, 0.750f})
        std::printf("%4.1f%%   Δp: %6.3lf%%\n", 100*q, 100.*kld_percentile(s.p_diff_values, q));
    std::printf("Median  Δp: %6.3lf%%\n", 100.*kld_median(s.p_diff_values));
    for (float q : {0.250f, 0.100f, 0.050f, 0.010f, 0.001f})
        std::printf("%4.1f%%   Δp: %6.3lf%%\n", 100*q, 100.*kld_percentile(s.p_diff_values, q));
    std::printf("Minimum Δp: %6.3lf%%\n", 100.*s.p_diff_values.front());
    auto mse = kld_mean_uncertainty(s.sum_p_diff2, s.sum_p_diff4, s.count);
    const double rms = std::sqrt(mse.first);
    std::printf("RMS Δp    : %6.3lf ± %5.3lf %%\n", 100*rms, 100*0.5/rms*mse.second);
    const double same = double(s.n_same_top)/s.count;
    std::printf("Same top p: %6.3lf ± %5.3lf %%\n", 100*same,
                100*std::sqrt(same*(1-same)/(s.count-1)));
}

} // namespace b70
