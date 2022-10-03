#include "assert.h"
#include "definitions.cuh"
#include <stdio.h>

// This function is an implementation of Algorithm AS 147:
//   available from http://ftp.uni-bayreuth.de/math/statlib/apstat/147
//
// Also refers to https://en.wikipedia.org/wiki/Chi-square_distribution
//   for what it does.
__device__ double deviceGammds(double x, double p) {
  double a;
  double arg;
  double c;
  double e = 1.0E-09;
  double f;
  // int ifault2;
  double uflo = 1.0E-37;
  double value;
  //
  //  Check the input.
  //
  if (x <= 0.0) {
    //*ifault = 1;
    value = 0.0;
    return value;
  }

  if (p <= 0.0) {
    //*ifault = 1;
    value = 0.0;
    return value;
  }
  //
  //  LGAMMA is the natural logarithm of the gamma function.
  //
  arg = p * log(x) - lgamma(p + 1.0) - x;

  if (arg < log(uflo)) {
    value = 0.0;
    //*ifault = 2;
    return value;
  }

  f = exp(arg);

  if (f == 0.0) {
    value = 0.0;
    //*ifault = 2;
    return value;
  }

  //*ifault = 0;
  //
  //  Series begins.
  //
  c = 1.0;
  value = 1.0;
  a = p;

  for (;;) {
    a = a + 1.0;
    c = c * x / a;
    value = value + c;

    if (c <= e * value) {
      break;
    }
  }

  value = value * f;

  return value;
  /*JR*/
}


__device__ double 
tally_contingency_table_resampled(
  int offset,
  short *resample_idx,       // the array indicates if corresponding samples is 
                           //    to be used (1) or not (0) in tallying contigency table.
  const int n_samples,     // the number of samples to process
                           //   n_samples_C1 or n_samples_C2 
  int *data,               // data stored in linear array
                           //   data_C1_linear or data_C2_linear
  int *row_ids,            // gene1 ids
  int *col_ids,            // gene2 ids

  int *dof,                // degree of freedom (computed in this function)
  int idx                  // local thread (threadIdx.x)
  ) {

  int con = 3;

  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  for (int k1 = 0; k1 < n_samples; k1++) {
    if (resample_idx[(blockIdx.x-offset) * n_samples + k1] == 0) {
      continue;
    }

    // place tally for each occurence in observed contingency table
    //
    // this will break if row_idx or col_idx goes out of bounds [0..2]
    int row_idx = data[(row_ids[idx] * n_samples) + k1] + 1;
    int col_idx = data[(col_ids[idx] * n_samples) + k1] + 1;
    tally[row_idx][col_idx]++;

  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //       idx, tally[0][0], tally[0][1], tally[0][2],
    //            tally[1][0], tally[1][1], tally[1][2],
    //            tally[2][0], tally[2][1], tally[2][2],
    //            ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }
  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  dof[threadIdx.x + blockDim.x * blockIdx.x] =
      ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
}

__device__ double 
tally_contingency_table_resampled_scalable(
  int offset,
  short *resample_idx,       // the array indicates if corresponding samples is 
                           //    to be used (1) or not (0) in tallying contigency table.
  const int n_samples,     // the number of samples to process
                           //   n_samples_C1 or n_samples_C2 
  int *data,               // data stored in linear array
                           //   data_C1_linear or data_C2_linear
  int *row_ids,            // gene1 ids
  int *col_ids,            // gene2 ids

  int *dof,                // degree of freedom (computed in this function)
  int localIdx,                  // local thread (threadIdx.x)
  int netId,
  int globalIdx
  ) {

  int skipper, con = 3;

  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  for (int k1 = 0; k1 < n_samples; k1++) {
    if (resample_idx[(netId-offset) * n_samples + k1] == 0) {
      continue;
    }

    // place tally for each occurence in observed contingency table
    //
    // this will break if row_idx or col_idx goes out of bounds [0..2]
    int row_idx = data[(row_ids[localIdx] * n_samples) + k1] + 1;
    int col_idx = data[(col_ids[localIdx] * n_samples) + k1] + 1;
    tally[row_idx][col_idx]++;

  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //       idx, tally[0][0], tally[0][1], tally[0][2],
    //            tally[1][0], tally[1][1], tally[1][2],
    //            tally[2][0], tally[2][1], tally[2][2],
    //            ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }
  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  dof[globalIdx] =
      ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
}

__device__ double sumrtime(const int offset, const int len, int *data, int *spc,
                           int *fr, int *dof, int idx) {

  int skipper, con = 3;
  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  skipper = blockIdx.x - offset;
  for (int k1 = 0; k1 < len; k1++) {
    if ((skipper != 0) && (k1 == skipper - 1)) {
      continue;
    }
    // place tally for each occurence in observed contingency table
    //
    // this will break if row_idx or col_idx goes out of bounds [0..2]
    int row_idx = data[(spc[idx] * len) + k1] + 1;
    int col_idx = data[(fr[idx] * len) + k1] + 1;
    tally[row_idx][col_idx]++;

//    if ((data[(spc[idx] * len) + k1] == -1) &&
//        (data[(fr[idx] * len) + k1] == -1)) {
//      tally[0][0]++;
//    } else if ((data[spc[idx] * len + k1] == -1) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[0][1]++;
//    } else if ((data[spc[idx] * len + k1] == -1) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[0][2]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == -1)) {
//      tally[1][0]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[1][1]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[1][2]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == -1)) {
//      tally[2][0]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[2][1]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[2][2]++;
//    }
  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //       idx, tally[0][0], tally[0][1], tally[0][2],
    //            tally[1][0], tally[1][1], tally[1][2],
    //            tally[2][0], tally[2][1], tally[2][2],
    //            ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }
  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  dof[threadIdx.x + blockDim.x * blockIdx.x] =
      ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
  // return tally[0][0];
}

__device__ double sumrtimeScalable(const int offset, const int len, int *data,
                                   int *spc, int *fr, int *dof, int idx,
                                   int netID, int globalIdx) {
  int skipper, con = 3;
  // contigency table observed
  int tally[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  // contigency table expected
  double expected[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // skipper = blockIdx.x - offset;
  skipper = netID - offset;
  for (int k1 = 0; k1 < len; k1++) {
    if ((skipper != 0) && (k1 == skipper - 1)) {
      continue;
    }

    // place tally for each occurence in observed contingency table
    //
    // this will break if row_idx or col_idx goes out of bounds [0..2]
    int row_idx = data[(spc[idx] * len) + k1] + 1;
    int col_idx = data[(fr[idx] * len) + k1] + 1;
    tally[row_idx][col_idx]++;

//    if ((data[(spc[idx] * len) + k1] == -1) &&
//        (data[(fr[idx] * len) + k1] == -1)) {
//      tally[0][0]++;
//    } else if ((data[spc[idx] * len + k1] == -1) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[0][1]++;
//    } else if ((data[spc[idx] * len + k1] == -1) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[0][2]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == -1)) {
//      tally[1][0]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[1][1]++;
//    } else if ((data[spc[idx] * len + k1] == 0) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[1][2]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == -1)) {
//      tally[2][0]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == 0)) {
//      tally[2][1]++;
//    } else if ((data[spc[idx] * len + k1] == 1) &&
//               (data[fr[idx] * len + k1] == 1)) {
//      tally[2][2]++;
//    }
  }

  // summation of rows and columns for chi squared table
  int ex[7] = {0, 0, 0, 0, 0, 0, 0};
  double yates = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if (c1 == 0) {
        ex[0] += tally[c][c1];
      } else if (c1 == 1) {
        ex[1] += tally[c][c1];
      } else if (c1 == 2) {
        ex[2] += tally[c][c1];
      }
    }
    for (int b = 0; b < con; b++) {
      if (b == 0) {
        ex[3] += tally[b][c];
      } else if (b == 1) {
        ex[4] += tally[b][c];
      } else if (b == 2) {
        ex[5] += tally[b][c];
      }
    }
  }

  if ((ex[0] + ex[1] + ex[2]) != (ex[3] + ex[4] + ex[5])) {
    printf("bad math!!!!!!!!");
  } else {
    ex[6] = ex[0] + ex[1] + ex[2];
    // printf("*** \n idx: %d \n %d %d %d \n %d %d %d \n %d %d %d \n %d %d %d %d
    // %d %d %d***",
    //                idx,
    //                tally[0][0], tally[0][1], tally[0][2],
    //                tally[1][0], tally[1][1], tally[1][2],
    //                tally[2][0], tally[2][1], tally[2][2],
    //                ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6], ex[7]);
  }

  double divisor = double(ex[6]);
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      expected[c][c1] = (double(ex[c1]) * double(ex[c + 3]) / divisor);
    }
  }

  // set use of yates correction if 1 cell < 5
  int flag = 0;
  for (int c = 0; c < con; c++) {
    for (int c1 = 0; c1 < con; c1++) {
      if ((expected[c][c1] < 5) && ((ex[c1]) && (ex[c + 3]))) {
        yates = .5;
        flag = 1;
        break;
      }
    }
    if (flag)
      break;
  }

  double chiSm = 0;
  int dofn = 0;
  int dofm = 0;

  // calculating chi squared sum
  for (int ii = 0; ii < 3; ii++) {
    if (ex[ii] == 0) {
      dofm++;
    }
    if (ex[ii + 3] == 0) {
      dofn++;
    }
    for (int jj = 0; jj < 3; jj++) {
      // save calculation time if not zero
      if ((ex[jj] * ex[ii + 3]) != 0) {
        chiSm += pow(abs(double(tally[ii][jj]) - expected[ii][jj]) - yates, 2) /
                 expected[ii][jj];
      }
    }
  }

  //
  // dof[threadIdx.x + blockDim.x*netID] = ((3 - dofm) - 1)*((3 - dofn) - 1);
  dof[globalIdx] = ((3 - dofm) - 1) * ((3 - dofn) - 1);

  return chiSm;
  // return tally[0][0];
}


// let's rename this function to "countStates" or "enumerateStates"(?)
__device__ void noStates(const int idx, const int noGenes, int samples1,
                         int samples2, int *data1, int *data2, int *out1,
                         int *out2) {
  int *dataIn;
  int samplesIn;
  int start;
  int stop;
  int retVal;

  if (idx < noGenes) {
    dataIn = data1;
    samplesIn = samples1;
    start = idx * samplesIn;
    stop = start + samplesIn;
    assert(stop <= noGenes * samples1);
  } else {
    dataIn = data2;
    samplesIn = samples2;
    start = (idx - noGenes) * samplesIn;
    stop = start + samplesIn;
    assert(stop <= noGenes * samples2);
  }

  unsigned short statedata1[3] = {0, 0, 0};

  for (int i = start; i < stop; i++) {

    // count the number of each state (-1, 0, 1)
    //   a.k.a., frequency
    statedata1[dataIn[i] + 1]++;

    // if (dataIn[i] == -1) {
    //   statedata1[0]++;
    // }
    // if (dataIn[i] == 0) {
    //   statedata1[1]++;
    // }
    // if (dataIn[i] == 1) {
    //   statedata1[2]++;
    // }
  }

  // count the number of observed states: 
  //    should be less than 3
  retVal = 3;  // default value
  for (int i = 0; i < 3; i++) {
    out2[idx * 3 + i] = 1;
    if (statedata1[i] == 0) {
      out2[idx * 3 + i] = 0;  // decrease if not observed at all

      retVal--;
    }
  }
  //if (retVal==0) printf ("EMPTY DATA\n");
  out1[idx] = retVal;
}


__global__ 
void __launch_bounds__(MAX_THREADS, 1) 
determineEdges_resampled(
  short *resample_idx_C1,    // the array indicates if corresponding samples is 
  short *resample_idx_C2,    //    to be used (1) or not (0) in tallying contigency table.
  const int n_genes, 
  const int n_samples_C1,
  const int n_samples_C2,
  int *data_C1_linear,
  int *data_C2_linear,
  int *priorMatrix,
  double alphaEdgePrior,
  double alphaEdge,
  bool flag_pAdjust,

  //
  // below are computed in this function
  //
  int *n_observed_states,  // the number of observed states in C1 and C2 (2 * num_genes)
                           //    the value should be 1, 2 or 3 
  int *freq_states,        // the frequency of each state in C1 and C2 (3 * 2 * num_genes)

  int *row_ids,            // row_ids and col_ids specify 
  int *col_ids,            //   how computed edges are stored in a linear array
                           //   (row_ids, col_ids) are also a pair of genes.
  int *dof_out,            // degree of freedom for ChiSq test
                           //   this is not used outside this kernel
                           // however, needed internally, and memory was allocated
                           //   to allow parallel computation and to carry the information 
                           //   between this kernel and sub-functions    

  int n_edges,
  int *edge_out            // edges (0: no edge, 1: edge)
  ) {

  // blockIdx.x 
  //   - the number of repeated resampling, 
  //     a.k.a., the number of networks to be generated/tried
  //    the number of unique networks will be less than n_repeats

  int index = threadIdx.x + blockDim.x * blockIdx.x; // global thread
  int tidx = threadIdx.x;                            // local thread (edge)
  int row = row_ids[tidx];
  int col = col_ids[tidx];

  //extern __shared__ int sharedMatrix[];

  //*(sharedMatrix + row * n_genes + col) = *(priorMatrix + row * n_genes + col);

  __syncthreads();

  double edgeVal = 0; // stores chisquared value then stores gammds value- hold
                      // edge value to see if edge

  if (index < n_genes * 2) {
    // freq_states: count frequency of each state (-1, 0, 1) and 
    // n_observed_states: count the observed states (should be 1, 2, 3)
    noStates(index, n_genes, n_samples_C1, n_samples_C2, data_C1_linear, data_C2_linear, n_observed_states, freq_states);
  }

  if (blockIdx.x <= n_samples_C1) {
    // edgeVal = sumrtime(0, n_samples_C1, data_C1_linear, row_ids, col_ids, dof_out, tidx);
    edgeVal = tally_contingency_table_resampled(0,resample_idx_C1, n_samples_C1, data_C1_linear, row_ids, col_ids, dof_out, tidx);
  } else {
    // edgeVal = sumrtime(n_samples_C1, n_samples_C2, data_C2_linear, row_ids, col_ids, dof_out, tidx);
    edgeVal = tally_contingency_table_resampled(n_samples_C1+1,resample_idx_C2, n_samples_C2, data_C2_linear, row_ids, col_ids, dof_out, tidx);
  }

  // edgeVal: p value of ChiSq (edge significance)
  edgeVal = 1.0 - deviceGammds(edgeVal / 2.0, ((double)dof_out[index]) / 2.0);

  // Bonferroni correction
  if (flag_pAdjust) {
    edgeVal = min(1.0, edgeVal * (((n_genes - 1) * n_genes) / 2));
  }

  if (edgeVal < alphaEdge || (*(priorMatrix + row * n_genes + col) == 1 &&
                              edgeVal < alphaEdgePrior)) {
    edge_out[index] = 1;
  } else {
    edge_out[index] = 0;
  }

}

__global__ 
void __launch_bounds__(MAX_THREADS, 1) 
determineEdges_resampled_scalable(
  short *resample_idx_C1,    // the array indicates if corresponding samples is 
  short *resample_idx_C2,    //    to be used (1) or not (0) in tallying contigency table.
  const int n_genes, 
  const int n_samples_C1,
  const int n_samples_C2,
  int *data_C1_linear,
  int *data_C2_linear,
  int *priorMatrix,
  double alphaEdgePrior,
  double alphaEdge,
  bool flag_pAdjust,

  //
  // below are computed in this function
  //
  int *n_observed_states,  // the number of observed states in C1 and C2 (2 * num_genes)
                           //    the value should be 1, 2 or 3 
  int *freq_states,        // the frequency of each state in C1 and C2 (3 * 2 * num_genes)

  int *row_ids,            // row_ids and col_ids specify 
  int *col_ids,            //   how computed edges are stored in a linear array
                           //   (row_ids, col_ids) are also a pair of genes.
  int *dof_out,            // degree of freedom for ChiSq test
                           //   this is not used outside this kernel
                           // however, needed internally, and memory was allocated
                           //   to allow parallel computation and to carry the information 
                           //   between this kernel and sub-functions    

  int n_edges,
  int *edge_out,            // edges (0: no edge, 1: edge)
  int BPN,      
  int TPB
  ) {

  // blockIdx.x 
  //   - the number of repeated resampling, 
  //     a.k.a., the number of networks to be generated/tried
  //    the number of unique networks will be less than n_repeats

  //extern __shared__ int sharedMatrix[];

  //*(sharedMatrix + row * n_genes + col) = *(priorMatrix + row * n_genes + col);

  __syncthreads();
  int netId = blockIdx.x / BPN;                                                 
  int localIdx = TPB * (blockIdx.x % BPN) + threadIdx.x;                        
  int globalIdx = localIdx + (netId * n_edges);                                       
                                                                                
  int row = row_ids[localIdx];                                                  
  int col = col_ids[localIdx];                                                     
  double edgeVal = 0; // stores chisquared value then stores gammds value- hold
                      // edge value to see if edge

  if (globalIdx < n_genes * 2) {
    // freq_states: count frequency of each state (-1, 0, 1) and 
    // n_observed_states: count the observed states (should be 1, 2, 3)
    noStates(globalIdx, n_genes, n_samples_C1, n_samples_C2, data_C1_linear, data_C2_linear, n_observed_states, freq_states);
  }

  if (netId <= n_samples_C1) {
    // edgeVal = sumrtime(0, n_samples_C1, data_C1_linear, row_ids, col_ids, dof_out, tidx);
    edgeVal = tally_contingency_table_resampled_scalable(0,resample_idx_C1, n_samples_C1, data_C1_linear, row_ids, col_ids, dof_out, localIdx, netId, globalIdx);
  } else {
    // edgeVal = sumrtime(n_samples_C1, n_samples_C2, data_C2_linear, row_ids, col_ids, dof_out, tidx);
    edgeVal = tally_contingency_table_resampled_scalable(n_samples_C1+1,resample_idx_C2, n_samples_C2, data_C2_linear, row_ids, col_ids, dof_out, localIdx, netId, globalIdx);
  }

  // edgeVal: p value of ChiSq (edge significance)
  edgeVal = 1.0 - deviceGammds(edgeVal / 2.0, ((double)dof_out[globalIdx]) / 2.0);

  // Bonferroni correction
  if (flag_pAdjust) {
    edgeVal = min(1.0, edgeVal * (((n_genes - 1) * n_genes) / 2));
  }

  if (edgeVal < alphaEdge || (*(priorMatrix + row * n_genes + col) == 1 &&
                              edgeVal < alphaEdgePrior)) {
    edge_out[globalIdx] = 1;
  } else {
    edge_out[globalIdx] = 0;
  }

}


__global__ void __launch_bounds__(MAX_THREADS,1) run2(const int noGenes, const int leng, const int lengb,
                     int *tary, int *taryb, int *spacr, int *ff, int *dofout,
                     int *ppn, int *stf, int *out, int c, int *priorMatrix,
                     double alphaEdgePrior, double alphaEdge,
                     bool flag_pAdjust) {

  int index = threadIdx.x + blockDim.x * blockIdx.x; // global thread
  int tdx = threadIdx.x;                             // local thread
  int row = spacr[tdx];
  int col = ff[tdx];

  extern __shared__ int sharedMatrix[];

  *(sharedMatrix + row * noGenes + col) = *(priorMatrix + row * noGenes + col);
  __syncthreads();
  double edgeVal = 0; // stores chisquared value then stores gammds value- hold
                      // edge value to see if edge

  if (index < noGenes * 2) {
    // creates contingency tables
    noStates(index, noGenes, leng, lengb, tary, taryb, ppn, stf);
  }

  if (blockIdx.x <= leng) {
    edgeVal = sumrtime(0, leng, tary, spacr, ff, dofout, tdx);
  } else {
    edgeVal = sumrtime(leng, lengb, taryb, spacr, ff, dofout, tdx);
  }

  // edgeVal: p value of ChiSq (edge significance)
  edgeVal = 1.0 - deviceGammds(edgeVal / 2.0, ((double)dofout[index]) / 2.0);

  // Bonferroni correction
  if (flag_pAdjust) {
    edgeVal = min(1.0, edgeVal * (((noGenes - 1) * noGenes) / 2));
  }

  if (edgeVal < alphaEdge || (*(sharedMatrix + row * noGenes + col) == 1 &&
                              edgeVal < alphaEdgePrior)) {
    out[index] = 1;
  } else {
    out[index] = 0;
  }
}

__global__ void __launch_bounds__(MAX_THREADS,1) run2Scalable(const int noGenes, const int leng, const int lengb,
                             int *tary, int *taryb, int *spacr, int *ff,
                             int *dofout, int *ppn, int *stf, int *out, int c,
                             int *priorMatrix, double alphaEdgePrior,
                             double alphaEdge, bool flag_pAdjust, int BPN,
                             int TPB) {
  int netId = blockIdx.x / BPN;
  int localIdx = TPB * (blockIdx.x % BPN) + threadIdx.x;
  int globalIdx = localIdx + (netId * c);

  if (localIdx < c) {
    int row = spacr[localIdx];
    int col = ff[localIdx];
    double edgeVal = 0.0;

    if (globalIdx < noGenes * 2) {
      noStates(globalIdx, noGenes, leng, lengb, tary, taryb, ppn, stf);
    }

    // do we need a __syncthreads here?
    if (netId <= leng) {
      edgeVal = sumrtimeScalable(0, leng, tary, spacr, ff, dofout, localIdx,
                                 netId, globalIdx);
    } else {
      edgeVal = sumrtimeScalable(leng, lengb, taryb, spacr, ff, dofout,
                                 localIdx, netId, globalIdx);
    }

    edgeVal =
        1 - deviceGammds(edgeVal / 2.0, ((double)dofout[globalIdx]) / 2.0);

    // Bonferroni correction
    if (flag_pAdjust) {
      edgeVal = min(1.0, edgeVal * (((noGenes - 1) * noGenes) / 2));
    }

    if (edgeVal < alphaEdge || (*(priorMatrix + row * noGenes + col) == 1 &&
                                edgeVal < alphaEdgePrior)) {
      out[globalIdx] = 1;
    } else {
      out[globalIdx] = 0;
    }
  }
}
