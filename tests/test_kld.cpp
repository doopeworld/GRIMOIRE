#include "b70/kld.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace {

constexpr int ctx = 8, vocab = 37, chunks = 2;
constexpr int scored = ctx - 1 - ctx/2;
constexpr int words = 2*((vocab + 1)/2) + 4;

std::vector<float> base_logits(int chunk, int row) {
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i) logits[i] = -2.f + 0.025f*i;
    logits[3] = 3.f + 0.1f*chunk + 0.03f*row;
    logits[5] = 1.f;
    return logits;
}

// Encode the reference independently, with the llama.cpp writer's float32
// scale/min and uint16 payload layout. These logits stay above its -16 cutoff.
std::vector<uint16_t> encode_row(const std::vector<float>& logits) {
    const float max = *std::max_element(logits.begin(), logits.end());
    const float min = *std::min_element(logits.begin(), logits.end());
    double sum = 0;
    for (float v : logits) sum += ::expf(v - max);
    const float scale = (max - min)/65535.f;
    const float min_log_prob = min - max - float(std::log(sum));
    std::vector<uint16_t> row(words, 0);
    std::memcpy(row.data(), &scale, 4);
    std::memcpy(row.data() + 2, &min_log_prob, 4);
    for (int i = 0; i < vocab; ++i)
        row[i+4] = uint16_t(std::lround((logits[i] - min)/scale));
    return row;
}

double expected_kld(const std::vector<float>& logits, const std::vector<uint16_t>& row) {
    float scale, min_log_prob;
    std::memcpy(&scale, row.data(), 4);
    std::memcpy(&min_log_prob, row.data()+2, 4);
    const double max = *std::max_element(logits.begin(), logits.end());
    double z = 0;
    for (float v : logits) z += std::exp(double(v)-max);
    const double logz = max + std::log(z);
    double result = 0;
    for (int i = 0; i < vocab; ++i) {
        const float log_base = scale*row[i+4] + min_log_prob;
        if (log_base > -16.f)
            result += std::exp(double(log_base))*(double(log_base) - double(logits[i]) + logz);
    }
    return result;
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("grimoire-kld-" + std::to_string(::getpid()) + ".bin");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("_logits_", 8);
        for (int32_t n : {ctx, vocab, chunks}) out.write(reinterpret_cast<const char*>(&n), 4);
        for (int32_t i = 0; i < chunks*ctx; ++i) out.write(reinterpret_cast<const char*>(&i), 4);
        for (int c = 0; c < chunks; ++c)
            for (int r = 0; r < scored; ++r) {
                auto row = encode_row(base_logits(c, r));
                out.write(reinterpret_cast<const char*>(row.data()), row.size()*2);
            }
        assert(out.good());
    }
    {
        b70::KldBaseReader reader(path.string());
        assert(reader.ctx() == ctx && reader.vocab() == vocab && reader.chunks() == chunks);
        assert(reader.scored_per_chunk() == scored && reader.row_words() == words);
        b70::KldStats equal, perturbed;
        std::vector<int32_t> tokens;
        std::vector<uint16_t> rows;
        for (int c = 0; c < chunks; ++c) {
            reader.read_chunk(c, tokens, rows);
            assert(tokens.front() == c*ctx && tokens.back() == c*ctx + ctx - 1);
            for (int r = 0; r < scored; ++r) {
                const auto base = base_logits(c, r);
                const auto expected_row = encode_row(base);
                const uint16_t* row = rows.data() + r*words;
                assert(std::equal(expected_row.begin(), expected_row.end(), row));
                const double same = equal.add(vocab, base.data(), row, tokens[ctx/2+r+1]).first;
                assert(std::abs(same) < 1e-4); // uint16 quantization error
                auto changed = base;
                changed[5] += 0.75f;
                const double hand = expected_kld(changed, expected_row);
                const double actual = perturbed.add(vocab, changed.data(), row, tokens[ctx/2+r+1]).first;
                assert(std::abs(actual - hand) < 2e-6);
                assert(actual > 0);
            }
        }
        assert(equal.count == chunks*scored && equal.n_same_top == equal.count);
        assert(perturbed.count == equal.count && perturbed.n_same_top == perturbed.count);
        assert(std::abs(equal.sum_kld/equal.count) < 1e-4);
        const auto mean = b70::kld_mean_uncertainty(perturbed.sum_kld, perturbed.sum_kld2, perturbed.count);
        double hand_mean = 0;
        for (float x : perturbed.kld_values) hand_mean += x;
        hand_mean /= perturbed.count;
        assert(std::abs(mean.first - hand_mean) < 1e-7);
        std::sort(perturbed.kld_values.begin(), perturbed.kld_values.end());
        const float expected_median = 0.5f*(perturbed.kld_values[2]+perturbed.kld_values[3]);
        assert(b70::kld_median(perturbed.kld_values) == expected_median);
        const float p90_index = 0.9f*(perturbed.kld_values.size()-1);
        const size_t lo = size_t(p90_index);
        const float p90 = (1-(p90_index-lo))*perturbed.kld_values[lo]
                        + (p90_index-lo)*perturbed.kld_values[lo+1];
        assert(b70::kld_percentile(perturbed.kld_values, 0.9f) == p90);
        // For observations 1..12, sum=78 and sum of squares=650.
        const auto uncertainty = b70::kld_mean_uncertainty(78, 650, 12);
        assert(std::abs(uncertainty.first - 6.5) < 1e-12);
        assert(std::abs(uncertainty.second - std::sqrt(13./12.)) < 1e-12);
        assert(std::abs(b70::kld_covariance(78, 156, 1300, 12) - 13./6.) < 1e-12);

        // The reference ignores p_log_base == -16, and both argmax scans
        // choose the first index on ties.
        std::vector<uint16_t> cutoff_row(4+3, 0);
        const float cutoff_scale = 0.5f, cutoff_min = -16.f;
        std::memcpy(cutoff_row.data(), &cutoff_scale, 4);
        std::memcpy(cutoff_row.data()+2, &cutoff_min, 4);
        cutoff_row[5] = cutoff_row[6] = 32;
        const float flat_logits[3] = {0, 0, 0};
        b70::KldStats cutoff;
        const double cutoff_kld = cutoff.add(3, flat_logits, cutoff_row.data(), 0).first;
        assert(std::abs(cutoff_kld - 2*std::log(3.)) < 2e-7);
        assert(cutoff.n_same_top == 0);
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.put('x');
    }
    bool rejected = false;
    try { b70::KldBaseReader trailing(path.string()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::filesystem::resize_file(path, std::filesystem::file_size(path)-2);
    rejected = false;
    try { b70::KldBaseReader truncated(path.string()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::filesystem::remove(path);
    std::cout << "KLD format, scoring, statistics, and truncation: PASS\n";
}
