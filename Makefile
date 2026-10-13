# muda - minimal CUDA-compatible runtime for the Tesla M40
#
# The build intentionally has no dependency on the CUDA toolkit or on the
# OpenCL development headers: libOpenCL is loaded with dlopen() at run time.

CC      ?= cc
CXX     ?= c++
AR      ?= ar
PYTHON  ?= python3
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter
CXXFLAGS ?= -O2 -g -std=gnu++17 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -D_GNU_SOURCE -Iinclude -Isrc
LDLIBS  += -ldl -lpthread -lm

BUILD   := build
LIB_SRC := src/muda.c src/driver_opencl.c src/muda_translate.c src/mudablas.c src/mudart.c
LIB_OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))

BINS    := $(BUILD)/mudainfo $(BUILD)/vector_add $(BUILD)/saxpy $(BUILD)/matmul \
           $(BUILD)/bench_gemm $(BUILD)/mlp $(BUILD)/test_muda \
           $(BUILD)/test_mudablas $(BUILD)/test_mudart $(BUILD)/test_translate \
           $(BUILD)/mudac $(BUILD)/vector_add_mudac $(BUILD)/test_mudac

.PHONY: all clean test python
all: $(BUILD)/libmuda.so $(BUILD)/libmuda.a $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@

$(BUILD)/libmuda.so: $(LIB_OBJ)
	$(CC) -shared -o $@ $^ $(LDLIBS)

$(BUILD)/libmuda.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

$(BUILD)/mudainfo: tools/mudainfo.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/vector_add: examples/vector_add.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/saxpy: examples/saxpy.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/matmul: examples/matmul.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/test_muda: tests/test_muda.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/bench_gemm: examples/bench_gemm.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/mlp: examples/mlp.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/test_mudablas: tests/test_mudablas.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/test_translate: tests/test_translate.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/test_mudart: tests/test_mudart.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/mudac: tools/mudac.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

$(BUILD)/vector_add_mudac.c: examples/vector_add.cu $(BUILD)/mudac | $(BUILD)
	$(BUILD)/mudac $< -o $@

$(BUILD)/vector_add_mudac: $(BUILD)/vector_add_mudac.c $(BUILD)/libmuda.a | $(BUILD)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -x c++ -o $@ $(BUILD)/vector_add_mudac.c -x none $(BUILD)/libmuda.a $(LDLIBS)

$(BUILD)/test_mudac.c: tests/test_mudac.cu $(BUILD)/mudac | $(BUILD)
	$(BUILD)/mudac $< -o $@

$(BUILD)/test_mudac: $(BUILD)/test_mudac.c $(BUILD)/libmuda.a | $(BUILD)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -x c++ -o $@ $(BUILD)/test_mudac.c -x none $(BUILD)/libmuda.a $(LDLIBS)

test: $(BUILD)/test_muda $(BUILD)/test_mudablas $(BUILD)/test_mudart $(BUILD)/test_translate $(BUILD)/vector_add_mudac $(BUILD)/test_mudac
	$(BUILD)/test_muda
	$(BUILD)/test_mudablas
	$(BUILD)/test_mudart
	$(BUILD)/test_translate
	$(BUILD)/vector_add_mudac
	$(BUILD)/test_mudac

python: all
	$(PYTHON) python/demo_kernel.py
	$(PYTHON) python/demo_matmul.py

clean:
	rm -rf $(BUILD)
