for view in {1..2}
do
    for thread in {2..8}
    do
    ./mandelbrot -t $thread -v $view
    done
done