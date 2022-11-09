clear
make clean
make
perf record -e cpu-cycles ./cg
perf report