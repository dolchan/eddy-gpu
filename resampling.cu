#include <math.h>

unsigned int get_proportional_resampling_size(const int n_resamplings, const int n_samples) {
  return(n_samples + n_resamplings * n_samples);
}

unsigned int get_leave_one_out_resampling_size(const int n_samples) {
  return(n_samples + n_samples * n_samples);
}

void construct_proportional_resampling_indices(const int n_resamplings, const int n_samples, const float prop, short *resampled) {
  unsigned int N = get_proportional_resampling_size(n_resamplings, n_samples);  // the first batch will be full data set
  unsigned int rand_threshold = round(RAND_MAX * (1.0 - prop));


  for (unsigned int i = 0; i < n_samples; i++) {
  	resampled[i] = 1;      // full data set
  }

  for (unsigned int i = n_samples; i < N; i++) {
    resampled[i] = rand() < rand_threshold ? 0 : 1;
  }  
}


void construct_leave_one_out_resampling_indices(const int n_samples, short *resampled) {
  // usngined long int is 0 to ~4.3G, 
  //   hence, probably not recommeded to use leave-one-out sampling 
  //   when n_samples > ~65k
  //
  // this is for both memory and compute-time consideration 

  unsigned int N = get_leave_one_out_resampling_size(n_samples);  // the first batch will be full data set

  for (unsigned int i = 0; i < n_samples; i++) {
    resampled[i] = 1;      // full data set
  }

  for (unsigned int i = n_samples; i < N; i++) {
    resampled[i] = 1;
  }

  // left out samples
  for (unsigned int i = 0; i < n_samples; i++) {
    // i = 0 -> (n_samples) + 0
    // i = 1 -> (n_samples) + n_samples*1 + 1
    // i = 2 -> (n_samples) + n_samples+2 + 2
    resampled[n_samples + n_samples*i + i] = 0;
  }
}