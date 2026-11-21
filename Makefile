NVCC      ?= nvcc
ARCH      ?= native
NVCCFLAGS ?= -O3 -lineinfo

gemm: main.cu kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ main.cu -lcublas

clean:
	rm -f gemm

.PHONY: clean
