# Host-only validation: no GPU, no oneAPI, just a C++17 compiler.
# Everything here can run on the Unraid box directly, or anywhere.
#
# GPU build: ./build_b70.sh   (needs oneAPI DPC++ + ocloc)
# Container: docker compose -f docker/docker-compose.yml build

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude

.PHONY: all test clean tools
all: test

bin/test_generation: tests/test_generation.cpp include/b70/generation.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_http: tests/test_http.cpp include/b70/http_request.hpp include/b70/json.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: test-correctness
test-correctness: bin/test_generation bin/test_http
	./bin/test_generation
	./bin/test_http

HOST_MODEL_SRC = src/quantize.cpp src/qwen35_loader.cpp src/safetensors.cpp src/native_model.cpp
tools: bin/grimoire-quantize bin/inspect_native_model

bin/grimoire-quantize: tools/b70_compile_model.cpp $(HOST_MODEL_SRC) $(wildcard include/b70/*.hpp)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

bin/inspect_native_model: tools/inspect_native_model.cpp src/native_model.cpp $(wildcard include/b70/*.hpp)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

bin/test_native_model: tests/test_native_model.cpp $(HOST_MODEL_SRC) $(wildcard include/b70/*.hpp) | bin/grimoire-quantize
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

.PHONY: test-native
test-native: bin/test_native_model bin/grimoire-quantize
	./bin/test_native_model ./bin/grimoire-quantize

bin/test_gptq: tests/test_gptq.cpp src/gptq.cpp src/quantize.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_formats: tests/test_formats.cpp src/quantize.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_attention: tests/test_attention.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_safetensors: tests/test_safetensors.cpp src/safetensors.cpp src/quantize.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_moe: tests/test_moe.cpp src/moe_ref.cpp src/quantize.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_deltanet: tests/test_deltanet.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_ops: tests/test_ops.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/test_k2_horizon: tests/test_k2_horizon.cpp include/b70/k2_horizon.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_k2_config: tests/test_k2_config.cpp $(HOST_MODEL_SRC) $(wildcard include/b70/*.hpp)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

bin/test_dflash2_selector: tests/test_dflash2_selector.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_dflash_config: tests/test_dflash_config.cpp include/b70/dflash_config.hpp include/b70/json.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/test_tokenizer: tests/test_tokenizer.cpp src/tokenizer.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

test: bin/test_formats bin/test_attention bin/test_safetensors bin/test_moe bin/test_deltanet bin/test_ops bin/test_tokenizer bin/test_gptq bin/test_k2_horizon bin/test_k2_config bin/test_dflash2_selector bin/test_dflash_config
	@echo "=========== formats ==========="; ./bin/test_formats
	@echo ""; echo "=========== attention ==========="; ./bin/test_attention
	@echo ""; echo "=========== safetensors ==========="; ./bin/test_safetensors
	@echo ""; echo "=========== moe ==========="; ./bin/test_moe
	@echo ""; echo "=========== deltanet ==========="; ./bin/test_deltanet
	@echo ""; echo "=========== ops ==========="; ./bin/test_ops
	@echo ""; echo "=========== tokenizer ==========="; ./bin/test_tokenizer
	@echo ""; echo "=========== gptq ==========="; ./bin/test_gptq
	@echo ""; echo "=========== k2-horizon ==========="; ./bin/test_k2_horizon
	@echo ""; echo "=========== k2-config ==========="; ./bin/test_k2_config
	@echo ""; echo "=========== dflash2-selector ==========="; ./bin/test_dflash2_selector
	@echo ""; echo "=========== dflash-config ==========="; ./bin/test_dflash_config

clean:
	rm -rf bin

# Standalone C++/SYCL profile: external bridges (including Torch/vLLM)
# are disabled at compile time. The default image is AOT for Battlemage G31.
# For compilation on a host without ocloc, use SYCL_TARGET=spir64.
SYCL_CXX ?= icpx
SYCL_TARGET ?= intel_gpu_bmg_g31
NATIVE_DIR = bin/native-$(SYCL_TARGET)
NATIVE_FLAGS = -fsycl -fsycl-targets=$(SYCL_TARGET) -O2 -std=c++20 \
 -fno-fast-math -ffp-contract=fast -fno-math-errno -DGRIMOIRE_NATIVE_ONLY -Iinclude -Isrc
NATIVE_SRC = grimoire qwen35_loader native_model safetensors quantize gptq \
 gemv_decode gemm_xmx attention deltanet moe_kernels moe_ref ops prefill tokenizer
NATIVE_OBJ = $(addprefix $(NATIVE_DIR)/,$(addsuffix .o,$(NATIVE_SRC)))

.PHONY: native
native: $(NATIVE_DIR)/grimoire $(NATIVE_DIR)/grimoire-server tools

$(NATIVE_DIR)/%.o: src/%.cpp
	@mkdir -p $(NATIVE_DIR)
	$(SYCL_CXX) $(NATIVE_FLAGS) -MMD -MP -c $< -o $@

$(NATIVE_DIR)/cli.o: tools/grimoire_main.cpp
	@mkdir -p $(NATIVE_DIR)
	$(SYCL_CXX) $(NATIVE_FLAGS) -MMD -MP -c $< -o $@

$(NATIVE_DIR)/server.o: tools/grimoire_server.cpp
	@mkdir -p $(NATIVE_DIR)
	$(SYCL_CXX) $(NATIVE_FLAGS) -MMD -MP -c $< -o $@

$(NATIVE_DIR)/grimoire: $(NATIVE_OBJ) $(NATIVE_DIR)/cli.o
	$(SYCL_CXX) $(NATIVE_FLAGS) $^ -ldl -lpthread -o $@

$(NATIVE_DIR)/grimoire-server: $(NATIVE_OBJ) $(NATIVE_DIR)/server.o
	$(SYCL_CXX) $(NATIVE_FLAGS) $^ -ldl -lpthread -o $@

-include $(wildcard $(NATIVE_DIR)/*.d)
