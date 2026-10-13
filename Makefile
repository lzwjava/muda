# muda - minimal CUDA-compatible runtime for the Tesla M40
#
# The build intentionally has no dependency on the CUDA toolkit or on the
# OpenCL development headers: libOpenCL is loaded with dlopen() at run time.

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -D_GNU_SOURCE -Iinclude -Isrc
LDLIBS  += -ldl -lpthread

BUILD   := build
LIB_SRC := src/muda.c src/driver_opencl.c src/muda_translate.c
LIB_OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))

BINS    := $(BUILD)/mudainfo $(BUILD)/vector_add $(BUILD)/test_muda

.PHONY: all clean test
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

$(BUILD)/test_muda: tests/test_muda.c $(BUILD)/libmuda.a | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(BUILD)/libmuda.a $(LDLIBS)

test: $(BUILD)/test_muda
	$(BUILD)/test_muda

clean:
	rm -rf $(BUILD)
