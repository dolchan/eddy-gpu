#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>


#define N 20
#define PROPORTION 0.9
#define MAX 100


int main(int argc, char* argv[]) {
  int N2 = N;
  float p = PROPORTION;

  if (argc > 1) {
    N2 = atoi(argv[1]);  
  }

  if (argc > 2) {
    p = atof(argv[2]);
  }

  short *rand_numbers = (short *)malloc(sizeof(short) * N2);

  int rand_threshold = round(RAND_MAX * (1.0 - p));

  srand((unsigned) time(0));

  for (int i = 0; i < N2; i++) {
    rand_numbers[i] = rand() < rand_threshold ? 0 : 1;
  }

  if (N2 <= 20) {
    for (int i = 0; i < N2; i++) {
      printf("%d\n", rand_numbers[i]);
    }
  } else {
    int nZeros = 0, nOnes = 0;

    for (int i = 0; i < N2; i++) {
      if (rand_numbers[i] == 0) 
        nZeros++;

      if (rand_numbers[i] == 1)
        nOnes++;
    }

    printf("0: %d, 1: %d, r = %f\n", nZeros, nOnes, 100.0*nOnes/N2);
  }

  free(rand_numbers);

  return 0;
}
