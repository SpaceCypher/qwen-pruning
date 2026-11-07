#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Intervention {
    std::array<unsigned long long, 32> masks{};
    bool trace = false;
    int hits = 0;
    int vector_layer = -1;
};

static bool intervene(ggml_tensor * tensor, bool ask, void * data) {
    auto & edit = *static_cast<Intervention *>(data);
    int layer = -1;
    int end = 0;
    const char * name = ggml_get_name(tensor);
    if ((std::sscanf(name, "ffn_swiglu-%d%n", &layer, &end) != 1 &&
         std::sscanf(name, "ffn_gate_par-%d%n", &layer, &end) != 1) ||
        name[end] || layer < 0 || layer >= 32 || tensor->ne[0] != 9216) {
        return false;
    }
    if (!edit.trace && !edit.masks[layer] && edit.vector_layer != layer) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        std::abort();
    }
    ++edit.hits;
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    if (edit.vector_layer == layer) {
        std::cout << "activation " << layer;
        for (auto it = values.end() - 9216; it != values.end(); ++it) {
            std::cout << ' ' << *it;
        }
        std::cout << '\n';
    }
    if (edit.trace) {
        const float * last = values.data() + values.size() - 9216;
        std::cout << "trace " << layer;
        for (int block = 0; block < 36; ++block) {
            double sum = 0;
            for (int j = 0; j < 256; ++j) {
                const double x = last[block * 256 + j];
                sum += x * x;
            }
            std::cout << ' ' << std::sqrt(sum);
        }
        std::cout << '\n';
    }
    for (size_t row = 0; row < values.size(); row += 9216) {
        for (int block = 0; block < 36; ++block) {
            if ((edit.masks[layer] >> block) & 1ULL) {
                std::fill_n(values.begin() + row + block * 256, 256, 0.0f);
            }
        }
    }
    if (edit.masks[layer]) {
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    std::vector<llama_token> ids(text.size() + 8);
    const int size = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, special);
    if (size < 1) {
        throw std::runtime_error("tokenization failed");
    }
    ids.resize(size);
    return ids;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        return 2;
    }
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        return 2;
    }
    Intervention edit;
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.cb_eval = intervene;
    params.cb_eval_user_data = &edit;
    auto * ctx = llama_init_from_model(model, params);
    if (!ctx) {
        return 2;
    }
    const auto * vocab = llama_model_get_vocab(model);
    std::array<llama_token, 10> choices{};
    for (int i = 0; i < 10; ++i) {
        const auto ids = tokenize(vocab, std::string(1, '0' + i), false);
        if (ids.size() != 1) {
            return 2;
        }
        choices[i] = ids.front();
    }
    std::cout << "ready" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.rfind("mask ", 0) == 0) {
            size_t offset = 5;
            for (auto & mask : edit.masks) {
                size_t length;
                mask = std::stoull(line.substr(offset), &length, 16);
                offset += length;
            }
            continue;
        }
        if (line.rfind("vector ", 0) == 0) {
            edit.vector_layer = std::stoi(line.substr(7));
            continue;
        }
        if (line == "trace on" || line == "trace off") {
            edit.trace = line == "trace on";
            continue;
        }
        llama_memory_clear(llama_get_memory(ctx), false);
        edit.hits = 0;
        const std::string prompt = "<|im_start|>user\n" + line +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        auto ids = tokenize(vocab, prompt, true);
        if (ids.size() > params.n_ctx || llama_decode(ctx, llama_batch_get_one(ids.data(), ids.size()))) {
            return 3;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int count = llama_vocab_n_tokens(vocab);
        const auto best = std::max_element(logits, logits + count);
        double sum = 0;
        for (int i = 0; i < count; ++i) {
            sum += std::exp(double(logits[i] - *best));
        }
        std::cout << "score " << edit.hits << ' ' << (best - logits);
        for (auto token : choices) {
            std::cout << ' ' << logits[token] - *best - std::log(sum);
        }
        std::cout << std::endl;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
