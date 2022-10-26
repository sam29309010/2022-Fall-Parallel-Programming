CXX=g++
CFLAGS=-std=c11 -g -Wall -g
LFALGS=-lpthread
SRC=pi.cpp
EXE=pi.out
OBJ=$(SRC:.c=.o)

all: $(EXE)

%.o: %.c
	$(CXX) $< -o $@ $(CFLAGS) $(LFALGS)

.PHONY: clean