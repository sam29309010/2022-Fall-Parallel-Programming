#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(){
    srand(time(NULL));
    long long int hit = 0, num_tosses = 100000000, toss;
    double x, y, dist, est_pi;
    for (toss=0; toss<num_tosses; toss++){
        x = ((double) rand() / RAND_MAX - 0.5) * 2;
        y = ((double) rand() / RAND_MAX - 0.5) * 2;
        dist = x * x + y * y;
        if (dist <= 1){
            hit++;
        }
    }
    est_pi = 4 * hit / ((double) num_tosses);
    cout << est_pi << endl;
}