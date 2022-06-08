#include <math.h>

void construct_resampling_indices(const int n_resamplings, const int n_samples, const float prop, short *resampled) {
  int N = n_samples + n_resamplings * n_samples;  // the first batch will be full data set
  int rand_threshold = round(RAND_MAX * (1.0 - prop));


  for (int i = 0; i < n_samples; i++) {
  	resampled[i] = 1;      // full data set
  }

  for (int i = n_samples; i < N; i++) {
    resampled[i] = rand() < rand_threshold ? 0 : 1;
  }  
}