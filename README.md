# eddy-gpu (master)
EDDY-GPU is a parallel implementation  of the EDDY (Evaluation of Differential DependencY) algorithm developed by the Biocomputing Lab at TGen and now mainted at the CRI Center for Computational Systems Biology at Prairie View A&M University. It is to be used with NVIDIA's CUDA API for GPUs. The original **EDDY** paper can be found at https://www.ncbi.nlm.nih.gov/pubmed/24500204. The EDDY website can be found at http://biocomputing.tgen.org/software/EDDY/index.html (old; removed).  

# Dependencies
eddy-gpu requires the Boost C++ library - boost/math/special_functions/ - for its ibetac function. Make sure Boost is installed before running eddy-gpu.

# Compiling
Compile:

```make```

On Texas Advanced Computing Center's (TACC) Maverick cluster compiling eddy-gpu is:

```make```

If nvcc is not listed as a command, you must load the CUDA module. For TACC Maverick, to load cuda version 7.5:

```module load cuda/7.5```

# Running
eddy-gpu has the following command line parameters:

```-d``` input data file

```-c``` class information file

```-g``` gene set list file

```-mp``` maximum number of parents for each node

```-pD``` pvalue threshold for divergence significance testing (DDN). [default = 0.10]

```-pE``` pvalue threshold for edge significance. [default = 0.05]

```-r``` number of permutations for statistical significance testing. [default = 100]

```-pw``` the prior knowledge weight

```-t``` theta for edge threshold.  This is to be deprecated.  Use -pE instead.

```-l``` lambda for edge threshold.  This is to be deprecated.  Use -pE instead.

Example command:
```./eddy -d input200.txt -c NKFB200.txt -g geneset40.txt -r 100 -mp 3 -pw .5 -p .05```

# Results
The Jensen-Shannon (JS) divergence score, p value, and number of unique networks are printed to the standard output stream.
If the analysis is deemed significant according to the predetermined p value, the following files will be created:
```geneset_file_name_BDEU_SCORES.txt``` contains the BDEU scores for each network for each class

```geneset_file_name_EdgeList.txt``` contains a list of edges with the class labeling and if the edge was determined from prior knowledge

```geneset_file_name_Networks.txt``` contains all of the edges for each unique network.

# Sponsor
The development of EDDY-GPU is partially funded by Compute the Cure|NVIDIA (https://blogs.nvidia.com/blog/2016/11/23/compute-the-cure-4/).

# Citation 
The manuscript describing EDDY-GPU has been presented as a short paper to PDP 2018 (https://ieeexplore.ieee.org/document/8374495).

Gil Speyer, Juan Rodriguez, Tomas Bencomo and Seungchan Kim, "GPU-accelerated differential dependency network analysis", [PDP 2018, Cambridge, UK, Mar 21-23, 2018.](https://ieeexplore.ieee.org/document/8374495)

Gil Speyer, Divya Mahendra, Hai J Tran, Jeff Kiefer, Stuart L Schreiber, Paul A Clemons, Harshil Dhruv, Michael Berens, Seungchan Kim, "Differential pathway dependency discovery associated with drug response across cancer cell lines", [Pac Symp Biocomput. 2017;22:497-508.  doi: 10.1142/9789813207813_0046] (https://pubmed.ncbi.nlm.nih.gov/27897001/)

Gil Speyer, Jeff Kiefer, Harshil Dhruv, Michael Berens, Seungchan Kim; "Knowledge-assisted approach to identify pathways with differential dependencies", [Pac Symp Biocomput. 2016;21:33-44] (https://pubmed.ncbi.nlm.nih.gov/26776171/)

