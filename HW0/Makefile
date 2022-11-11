CXX=g++
FLAG=-Wall -g
SRC=pi.cpp
EXE=pi.out

all: ${EXE}

${EXE}:${SRC}
	${CXX} ${SRC} -o ${EXE}

.PHONY: clean

clean:
	rm -rf ${EXE}