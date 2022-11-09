clear
make clean
make
perf record -e cycles,instructions,cache-misses ./bfs_grader ../data
perf report