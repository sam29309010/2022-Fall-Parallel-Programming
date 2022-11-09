repeat () { 
   for i in {1..10}
   do
    ./cg
   done
}

clear
make clean
make
time repeat

