/* random3.cu */

#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define nBlocks 2
#define nThreads 3
#define N (nBlocks*nThreads)


#define MAX 100

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, unsigned long long *seeds, 
                     unsigned long long *sequences, curandState_t* states) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    seeds[idx] = seed;
    sequences[idx] = blockIdx.x;

    /* we have to initialize the state */
    curand_init(seeds[idx], /* the seed can be the same for each core, here we pass the time in from the CPU */
                sequences[idx], /* the sequence number should be different for each core (unless you want all
                                   cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[idx]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    numbers[idx] = curand(&(states[idx])) % MAX;
}

int main(int argc, char* argv[]) {
    int nRandomNumbers;

    if (argc > 1) {
        nRandomNumbers = atoi(argv[1]);
    } else {
        nRandomNumbers = 20;
    }


    /* CUDA's random number library uses curandState_t to keep track of the seed value
       we will store a random state for every thread  */
    curandState_t* states;
    unsigned long long *seeds;
    unsigned long long *sequences;

    curandState_t* h_states;
    unsigned long long *h_seeds;
    unsigned long long *h_sequences;


    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &seeds, N * sizeof(unsigned long long));
    cudaMalloc((void**) &sequences, N * sizeof(unsigned long long));
    cudaMalloc((void**) &states, N * sizeof(curandState_t));

    /* invoke the GPU to initialize all of the random states */
    init<<<nBlocks, nThreads>>>(0, seeds, sequences, states);

    h_seeds = (unsigned long long *)malloc(N * sizeof(unsigned long long));
    h_sequences = (unsigned long long *)malloc(N * sizeof(unsigned long long));
    h_states = (curandState_t *)malloc(N * sizeof(curandState_t));

    cudaMemcpy(h_seeds, seeds, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sequences, sequences, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_states, states, N * sizeof(curandState_t), cudaMemcpyDeviceToHost);


    printf("seed, sequence, curandState_t\n");
    for (int i = 0; i < nBlocks; i++) {
        printf("Block [%d]\n", i);
        for (int j = 0; j < nThreads; j++) {
            int idx = i * nThreads + j; 
            printf("\tThread [%u] (%llu, %llu), %u - ", j, h_seeds[idx], h_sequences[idx], h_states[idx].d);
            for (int k = 0; k < 5; k++) {
                printf("%d, ", h_states[idx].v[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* allocate an array of unsigned ints on the CPU and GPU */
    unsigned int* cpu_nums;
    unsigned int* gpu_nums;

    cpu_nums = (unsigned int*)malloc(nRandomNumbers * N * sizeof(unsigned int));
    cudaMalloc((void**) &gpu_nums, nRandomNumbers * N * sizeof(unsigned int));

    cudaDeviceSynchronize();

    /* invoke the kernel to get some random numbers */
    for (int i = 0; i < nRandomNumbers; i++)
      randoms<<<nBlocks, nThreads>>>(states, gpu_nums + N*i);

    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("ERROR... %s\n", cudaGetErrorString(err));
    }

    /* copy the random numbers back */
    cudaMemcpy(cpu_nums, gpu_nums, nRandomNumbers * N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* print them out */
    for (int i = 0; i < nBlocks; i++) {
        printf("Block [%d]\n", i);        
        for (int j = 0; j < nThreads; j++) {
            printf("\tThread [%d]: ", j);
            for (int k = 0; k < nRandomNumbers; k++) {
                printf("%u ", (cpu_nums + k*N)[i*nThreads + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* free the memory we allocated for the states and numbers */
    cudaFree(states);
    cudaFree(seeds);
    cudaFree(sequences);
    cudaFree(gpu_nums);

    free(cpu_nums);
    free(h_states);
    free(h_seeds);
    free(h_sequences);

    return 0;
}

