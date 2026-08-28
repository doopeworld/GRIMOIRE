# Host-only validation: no GPU, no oneAPI, just a C++17 compiler.
# Everything here can run on the Unraid box directly, or anywhere.
#
# GPU build: ./build_b70.sh   (needs oneAPI DPC++ + ocloc)
# Container: docker compose -f docker/docker-compose.yml build

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude

.PHONY: all test clean
all: test

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

bin/test_tokenizer: tests/test_tokenizer.cpp src/tokenizer.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@

test: bin/test_formats bin/test_attention bin/test_safetensors bin/test_moe bin/test_deltanet bin/test_ops bin/test_tokenizer bin/test_gptq
	@echo "=========== formats ==========="; ./bin/test_formats
	@echo ""; echo "=========== attention ==========="; ./bin/test_attention
	@echo ""; echo "=========== safetensors ==========="; ./bin/test_safetensors
	@echo ""; echo "=========== moe ==========="; ./bin/test_moe
	@echo ""; echo "=========== deltanet ==========="; ./bin/test_deltanet
	@echo ""; echo "=========== ops ==========="; ./bin/test_ops
	@echo ""; echo "=========== tokenizer ==========="; ./bin/test_tokenizer
	@echo ""; echo "=========== gptq ==========="; ./bin/test_gptq

clean:
	rm -rf bin
