CXX=gcc
CFLAGS=-g -Wall -O3 -mavx2 
IFLAGS=-ISIMDxorshift/include
LFALGS=-lpthread
# TBD debug implementation
# TBD make sure the avx2 is used
SIMD_RAN_PATH=./SIMDxorshift/src/
SRC=pi.c $(SIMD_RAN_PATH)/simdxorshift128plus.c $(SIMD_RAN_PATH)/xorshift128plus.c
OBJ=${SRC:.c=.o}
EXE=pi.out

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LFALGS) $(IFLAGS)

%.o: %.c
	$(CXX) -c $^ -o $@ $(CFLAGS) $(LFALGS) $(IFLAGS)
	$(CXX) -S $^ $(CFLAGS) $(LFALGS) $(IFLAGS)

.PHONY: clean

clean:
	rm -rf *.o *.s $(SIMD_RAN_PATH)/*.o $(SIMD_RAN_PATH)/*.s *.out