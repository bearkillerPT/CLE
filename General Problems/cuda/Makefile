

all: RowDet

%: %.cu
	nvcc -O2 -Wno-deprecated-gpu-targets -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
