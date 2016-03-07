// PMA
/*
 If use please cite:
 @inproceedings{Nobari:2012,
  author    = {Sadegh Nobari and
               Thanh{-}Tung Cao and
               Panagiotis Karras and
               St{\'{e}}phane Bressan},
  title     = {Scalable parallel minimum spanning forest computation},
  booktitle = {Proceedings of the 17th {ACM} {SIGPLAN} Symposium on Principles and
               Practice of Parallel Programming, {PPOPP} 2012, New Orleans, LA, USA,
               February 25-29, 2012},
  pages     = {205--214},
  year      = {2012},
  crossref  = {DBLP:conf/ppopp/2012},
  url       = {http://doi.acm.org/10.1145/2145816.2145842},
  doi       = {10.1145/2145816.2145842},
  timestamp = {Tue, 22 May 2012 15:24:56 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/ppopp/NobariCKB12},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
*/

#define OUTPUT

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include "main.h"

#include <exception>

//#include "Timer.h"


#define PRIM_MIN            1
#define	PRIM_RADIXSORT		2
#define PRIM_HYBRID         3

#define HYBRID_MINVERT      10000
#define HYBRID_MINDEG       1000

#define REDUNDANT_MIN_E     1000

//Global Variables
int runs					= 10;		// Repeat how many times
unsigned int maxQueueSize	= 100; //2; //70000000; // Maximum total queue size

int maxPrimGrid                 = 512; 
int primBlock                   = 256; 

int dimGrid						= 512; 
int dimBlock					= 256; 

int primAlgo				=  PRIM_HYBRID;		// Choose the algorithm
//int primAlgo				=  PRIM_MIN;		// Choose the algorithm
//int primAlgo				=  PRIM_RADIXSORT;		// Choose the algorithm

// Input format is Edge list (true) or Vertex list (false)
bool edgelist			= false;	

// Input filename, set by -i argument.
bool inputBinary		= true;		// Use binary input or text?
string fname			= "Eastern.gr";

// Size in bytes
VTYPE SizeVV,			// VertexNum * VTYPE
	  SizeEV,			// EdgeNum * VTYPE
	  SizeEW;			// EdgeNum * WTYPE

// No. of vertices and edges
VTYPE VertexNum, EdgeNum;

WTYPE *EW;					// Edge weight
VTYPE *V1, *V2, *EJ;		// Starting vertices, ending vertices, destionations
VTYPE *ESIndex;				// Starting index of each vertex's edge list

VTYPE noComp; 

// Input parsing progress
char percent[7] = "0.0%%";


bool CPU=false;		//Run the sequentials?

// Update cycle
int refcycle	= 10000;

// Result
WTYPE totalWeight, mstEdgeNo; 

// Timing
double stepTime[10]; 

// Swap 2 VTYPE pointers
void swapVTYPE(VTYPE **a, VTYPE **b) {
    VTYPE *tmp; 

    tmp = *a; 
    *a = *b; 
    *b = tmp; 
}

// Initialize CUDA
#ifdef __DEVICE_EMULATION__
bool InitCUDA(void){return true;}
#else
bool InitCUDA(void)
{
    int count = 0;
    int i = 0;
    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }
    cudaSetDevice(i);

    printf("CUDA initialized.\n");

	return true;
}
#endif

void usage(char *program_name) {

    printf("\tUsage: %s [-v] [-h] [-a <int>] [-i <filename>]\n", program_name);
    printf("   -h               Print this help menu.\n");
    printf("   -a               Algorithm 1:MinPARMSF 2:SortPARMSF 3:HPARMSF\n");
	printf("   -t               # of threads\n");
	printf("   -b               # of block\n");
	printf("   -Q               Max Q Size\n");
	printf("   -r               Max # of runs\n");
	printf("   -c               run the sequentials on CPU as well\n");
    printf("   -i <filename>    A Graph from a file followed by the filename(full Path)\n");
    printf("\n-a 3 -i 'c:\\data\\DIMACS\\ONY.gr' -t 32 -b 14 -Q 6 -r 10");


}
void readTextInput() {
    //Start of reading
	cout << endl; 
    cout << "Reading input text file..." << endl;

	ifstream inf; 

    inf.open(fname.c_str());

    if (!inf) {
		cout << "Cannot open input file" << endl; 
		exit(1);
	}

    inf >> VertexNum >> EdgeNum;

    SizeVV = sizeof(VTYPE) * (VertexNum + 1);
    SizeEV = sizeof(VTYPE) * (EdgeNum);
    SizeEW = sizeof(WTYPE) * (EdgeNum);

    cout << "ES:" << SizeVV / 1000.0 << "KB, EJ:" << SizeEV / 1000.0
		 << "KB, EW:" << SizeEW / 1000.0 << "KB" << endl;

    ESIndex = (VTYPE *) malloc(SizeVV);
    EJ = (VTYPE *) malloc(SizeEV);
    EW = (WTYPE *) malloc(SizeEW);

	int edgeCount = 0; 

    for(VTYPE i = 0; i < VertexNum; i++)
    {
		// Display parsing progress
        if (i % refcycle == 0) {
            sprintf(percent, "%3.2f%%", ((double) i / VertexNum) * 100.0);
            printf("%s\r", percent);
        }

	    ESIndex[i] = edgeCount;

		// Read one edge first. Each vertex must have a non-zero degree
        inf >> EJ[edgeCount] >> EW[edgeCount];
		edgeCount++; 

		// Read until end of line
        while(inf.peek() != '\n')	
        {
            inf >> EJ[edgeCount] >> EW[edgeCount];
			edgeCount++; 
        }
    } 

    inf.close();

    cout << "Reading Completed." << endl;
	cout << endl; 
	cout << "No. vertices: " << VertexNum << endl; 
	cout << "No. edges: " << EdgeNum << endl; 
	cout << endl; 
}

void readBinaryInput() {
    //Start of reading
	cout << endl; 
    cout << "Reading input binary file..." << endl;

	FILE *fin = fopen(fname.c_str(), "rb");

    if (fin == 0) {
		cout << "Cannot open input file" << endl; 
		exit(1);
	}

	fread(&VertexNum, sizeof(VTYPE), 1, fin);
	fread(&EdgeNum, sizeof(VTYPE), 1, fin);

    SizeVV = sizeof(VTYPE) * (VertexNum + 1);
    SizeEV = sizeof(VTYPE) * (EdgeNum);
    SizeEW = sizeof(WTYPE) * (EdgeNum);

    cout << "ES:" << SizeVV / 1000.0 << "KB, EJ:" << SizeEV / 1000.0
		 << "KB, EW:" << SizeEW / 1000.0 << "KB" << endl;

    ESIndex = (VTYPE *) malloc(SizeVV);
    EJ = (VTYPE *) malloc(SizeEV);
    EW = (WTYPE *) malloc(SizeEW);

	fread(ESIndex, sizeof(VTYPE), VertexNum + 1, fin);
	fread(EJ, sizeof(VTYPE), EdgeNum, fin);
	fread(EW, sizeof(WTYPE), EdgeNum, fin);

	fclose(fin); 

    cout << "Reading Completed." << endl;
	cout << endl; 
	cout << "No. vertices: " << VertexNum << endl; 
	cout << "No. edges: " << EdgeNum << endl; 
	cout << endl; 
}

//Parsing Arguments
void parse_args(int argc, char **argv) {

    edgelist=false;
    int x;

    for ( x= 1; x < argc; ++x)
    {
        switch (argv[x][1])
        {
        case 'h':
            usage(argv[0]);
            exit(1);
            break;
		case 'c':
			CPU = true;
            break;
        case 'e':
            edgelist=true;
            break;

        case 'i':
            if (++x < argc)
            {
                fname=argv[x];
            }
            else
            {
                fprintf(stderr, "Error: Invalid argument, %s", argv[x-1]);
                usage(argv[0]);
            }
            break;
		case 'a':       /* Algorithm 1:MinPARMSF 2:SortPARMSF 3:HPARMSF */
			sscanf(argv[++x], "%u", &primAlgo);
            break;
		case 'r':       /* Max # of runs */
			sscanf(argv[++x], "%u", &runs);
            break;
		case 'Q':       /* Max Q Size */
			sscanf(argv[++x], "%u", &maxQueueSize);
            break;
        default:
            fprintf(stderr, "Error: Invalid command line parameter, %c\n", argv[x][1]);
            usage(argv[0]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! PMST
////////////////////////////////////////////////////////////////////////////////
// Data
VTYPE *d_EI, *EI;					// We compute this in CUDA and copy back to CPU to use
VTYPE *d_EJ, *d_EW, *d_ESIndex;		// The graph

VTYPE *d_VisitList, *d_Queue, *d_QueueI;	// Prim
VTYPE *d_tag;						// To compute new ID for each component
VTYPE *d_EkeyW, *d_EvalW;						//To do the sortings in segmented version as a temp array
VTYPE *d_Ekey,*d_Eval;				// To sort the new edge list by new starting vertex
VTYPE *d_ESIndexNew, *d_EWNew,
      *d_EJNew, *d_EINew,
	  *d_NEWEdgeNum;				// New graph for the next iteration

#ifdef OUTPUT
VTYPE *parent; 
BYTE *EIFlag, *d_EIFlag;			// Mark edges chosen in the MST
VTYPE *d_EOrg, *d_EOrgNew;			// Keep track of original edge index of each edge
#endif

bool *flag, *d_flag;						// A flag to signal the completion of the MST construction
int iter;								// Number of iterations executed

VTYPE VerticesPerThread;
VTYPE orgEdgeNum, orgVertexNum;			// Store the original number of vertices/edges
VTYPE QueueSize;						// Total queue size

VTYPE CompNum;

// CUDPP
CUDPPHandle planSort, planScan, planRSort;
CUDPPResult resultSort, resultScan, resultRSort;  
CUDPPConfiguration configSort, configScan, configRSort;
CUDPPHandle planCompact; 
CUDPPConfiguration configCompact; 

// Timing
unsigned int timerTotal, timerInit, timerAlloc, timerCopy, timerExec, timerDup, timerVPrim, timerMerge,
 timerBoruvka, timerPrim, timerKruskal, timerGConst, timerTest, timerParallel, timerIter;
    
// Initialize the parameters for PMST
void initializeParameters() 
{
	// Store the original number of vertices/edges
    orgEdgeNum = EdgeNum; 
    orgVertexNum = VertexNum; 

	// Compute the total queue size
	// If it's too big, limit the queue for each thread 
	// (currently use 100, should use maxQueueSize
    QueueSize = maxQueueSize * maxPrimGrid * primBlock;

 	// CUDPP
    // Sort Ekey and Eval
    configSort.algorithm = CUDPP_SORT_RADIX;
    configSort.datatype = CUDPP_UINT;
    configSort.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

    // Compact edges
    configCompact.algorithm = CUDPP_COMPACT; 
    configCompact.datatype = CUDPP_UINT;
    configCompact.options = CUDPP_OPTION_FORWARD;

	// CUDPP
    // Radix Sort
	configRSort.algorithm = CUDPP_SORT_RADIX;
    configRSort.datatype = CUDPP_UINT;
    configRSort.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

    // Exclusive scan of tags
    configScan.algorithm = CUDPP_SCAN;
    configScan.datatype = CUDPP_UINT;
    configScan.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    configScan.op = CUDPP_ADD;
}

void allocateMemory() 
{
    // Allocate Host memory
    cutilSafeCall( cudaHostAlloc( (void**) &EI, EdgeNum * sizeof(VTYPE), 0));
    cutilSafeCall( cudaHostAlloc( (void**) &flag, sizeof(bool), 0));

#ifdef OUTPUT
    cutilSafeCall( cudaHostAlloc( (void**) &EIFlag, EdgeNum * sizeof(VTYPE), 0));
    cutilSafeCall( cudaHostAlloc( (void**) &parent, VertexNum * sizeof(VTYPE), 0));
#endif
 
    // Allocate Device memory
    cutilSafeCall( cudaMalloc( (void**) &d_EI, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_EJ, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_EW, SizeEW));
    cutilSafeCall( cudaMalloc( (void**) &d_ESIndex, SizeVV));
    cutilSafeCall( cudaMalloc( (void**) &d_flag, sizeof(bool)));
	cutilSafeCall( cudaMalloc( (void**) &d_VisitList, SizeVV));
    cutilSafeCall( cudaMalloc( (void**) &d_Queue, QueueSize * sizeof(VTYPE)));

    cutilSafeCall( cudaMalloc( (void**) &d_QueueI, QueueSize * sizeof(VTYPE)));
    cutilSafeCall( cudaMalloc( (void**) &d_tag, SizeVV));
    cutilSafeCall( cudaMalloc( (void**) &d_Ekey, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_Eval, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_NEWEdgeNum, sizeof(VTYPE)));
    cutilSafeCall( cudaMalloc( (void**) &d_EINew, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_EJNew, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_EWNew, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_ESIndexNew, SizeVV));

#ifdef OUTPUT
    cutilSafeCall( cudaMalloc( (void**) &d_EIFlag, EdgeNum * sizeof(BYTE)));
    cutilSafeCall( cudaMalloc( (void**) &d_EOrgNew, SizeEV));
    cutilSafeCall( cudaMalloc( (void**) &d_EOrg, SizeEV));        
#endif

    // CUDPP, freaking slow, put it out of the timing
    cudppPlan(&planSort, configSort, EdgeNum, 1, 0);	
    cudppPlan(&planScan, configScan, VertexNum+1, 1, 0);  
    cudppPlan(&planRSort, configRSort, EdgeNum, 1, 0);	
    cudppPlan(&planCompact, configCompact, EdgeNum, 1, 0); 
}

void releaseMemory() 
{
    CUDA_SAFE_CALL(cudaFree(d_EI));
    CUDA_SAFE_CALL(cudaFree(d_EJ));
    CUDA_SAFE_CALL(cudaFree(d_EW));
    CUDA_SAFE_CALL(cudaFree(d_ESIndex));
    CUDA_SAFE_CALL(cudaFree(d_flag));
    CUDA_SAFE_CALL(cudaFree(d_VisitList));
	CUDA_SAFE_CALL(cudaFree(d_Queue));

	CUDA_SAFE_CALL(cudaFree(d_QueueI));
    CUDA_SAFE_CALL(cudaFree(d_tag));
    CUDA_SAFE_CALL(cudaFree(d_Ekey));
    CUDA_SAFE_CALL(cudaFree(d_Eval));
    CUDA_SAFE_CALL(cudaFree(d_NEWEdgeNum));
    CUDA_SAFE_CALL(cudaFree(d_ESIndexNew));
    CUDA_SAFE_CALL(cudaFree(d_EWNew));
    CUDA_SAFE_CALL(cudaFree(d_EJNew));
    CUDA_SAFE_CALL(cudaFree(d_EINew));
 
#ifdef OUTPUT
    CUDA_SAFE_CALL(cudaFree(d_EIFlag));
    CUDA_SAFE_CALL(cudaFree(d_EOrgNew));
    CUDA_SAFE_CALL(cudaFree(d_EOrg));      
#endif
 
	// Release Host memory
    CUDA_SAFE_CALL(cudaFreeHost(flag));
    CUDA_SAFE_CALL(cudaFreeHost(EI));

#ifdef OUTPUT
    CUDA_SAFE_CALL(cudaFreeHost(EIFlag));
    CUDA_SAFE_CALL(cudaFreeHost(parent));
#endif

	// CUDPP
	cudppDestroyPlan(planSort); 
	cudppDestroyPlan(planScan); 
	cudppDestroyPlan(planRSort);
    cudppDestroyPlan(planCompact); 
}

// Copy input graph from CPU to GPU
void copyInput() 
{
    // Transfer the graph to GPU
    cutilSafeCall( cudaMemcpy( d_EJ, EJ, SizeEV, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_EW, EW, SizeEW, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_ESIndex, ESIndex, SizeVV, cudaMemcpyHostToDevice) );
}

// Initialize the GPU datastructure for PMST
void initializeGPUData() {
#ifdef OUTPUT
    // Initialize the original edge index
    InitializeIndexInvoker(dimGrid, dimBlock, d_EOrg, EdgeNum); 

	// Clear the marker array
    cutilSafeCall( cudaMemset(d_EIFlag, 0, EdgeNum * sizeof(BYTE)));
#endif

    // Initialize EI, the starting vertex of each edge
    InitializeInvoker(dimGrid, dimBlock, d_EI, d_ESIndex, VertexNum);

	// Get EI back to CPU to use later
    cutilSafeCall( cudaMemcpy(EI, d_EI, SizeEV, cudaMemcpyDeviceToHost) );
}

// Print result and logging
void reportResult()
{
    cout << "Iterations=" << iter << " Edges=" << mstEdgeNo << " Weight=" 
		 << totalWeight << " Time: " << cutGetTimerValue(timerTotal) << endl;
	cout << "... Init: " << cutGetTimerValue(timerInit) 
		<< " Alloc: " << cutGetTimerValue(timerAlloc)
		<< " Copy: " << cutGetTimerValue(timerCopy)
		<< " Exec: " << cutGetTimerValue(timerExec)
		<< " Dup: " << cutGetTimerValue(timerDup) << endl; 
}

void sortEdgeList() 
{
    //Execute the radix sort

    int VBits = (int)((log((double) VertexNum) / log(2.0))) + 1;
    int EBits = 32;
	
    d_EkeyW = d_Ekey; 
    d_EvalW = d_Eval; 

    InitializeIndexInvoker(dimGrid, dimBlock, d_EvalW, EdgeNum); 

	// Sort by weight
    cutilSafeCall( cudaMemcpy( d_EkeyW, d_EW, EdgeNum * sizeof(WTYPE), cudaMemcpyDeviceToDevice) );

    cudppSort(planRSort, d_EkeyW, (void*)d_EvalW, EBits, EdgeNum);
 
    //Now replacing the edge list with respect to the sort
    GatherInvoker(dimGrid, dimBlock, d_EI, d_EINew, d_EvalW, EdgeNum);
	
	// Prepare
    swapVTYPE( &d_EI , &d_EINew); 

	cudppSort(planRSort, d_EI, (void*)d_EvalW, VBits, EdgeNum);
	
	//Now replacing the edge list with respect to the sort

#ifdef OUTPUT
    GatherEsInvoker(dimGrid, dimBlock, d_EvalW, 0, d_EW, d_EWNew, d_EI, 
		d_EJ, d_EJNew, d_EOrg, d_EOrgNew, EdgeNum, VertexNum);
#else
    GatherEsInvoker(dimGrid, dimBlock, d_Eval, 0, d_EW, d_EWNew, d_EI, 
		d_EJ, d_EJNew, 0, 0, EdgeNum, VertexNum);
#endif

    swapVTYPE(&d_EW, &d_EWNew); 
    swapVTYPE(&d_EJ, &d_EJNew); 

#ifdef OUTPUT
    swapVTYPE(&d_EOrg, &d_EOrgNew); 
#endif
}

void runPrimRadixSort(){    
	int primGrid = MIN(maxPrimGrid, (VertexNum / primBlock / maxQueueSize) + 1); 

    //printf("Total threads: %i\n", primGrid * primBlock); 

	// Execute The Visiting Prim
#ifdef OUTPUT
    VisitingPrimInvoker(primGrid, primBlock, d_VisitList, d_ESIndex, d_EW, d_EJ,
		d_Queue, d_QueueI, maxQueueSize, VertexNum, d_EIFlag, d_EOrg);
#else
    VisitingPrimInvoker(primGrid, primBlock, d_VisitList, d_ESIndex, d_EW, d_EJ,
		d_Queue, d_QueueI, maxQueueSize, VertexNum, 0, 0);
#endif
}

void runPrimMin(){    
    int primGrid = MIN(maxPrimGrid, (VertexNum / primBlock / maxQueueSize) + 1); 

    //printf("Total threads: %i\n", primGrid * primBlock); 

	// Execute The Visiting Prim
#ifdef OUTPUT
    VisitingPrimNoSortInvoker(primGrid, primBlock, d_VisitList, d_ESIndex, d_EW, d_EJ,
		d_Queue, d_QueueI, maxQueueSize, VertexNum, d_EIFlag, d_EOrg);
#else
    VisitingPrimNoSortInvoker(primGrid, primBlock, d_VisitList, d_ESIndex, d_EW, d_EJ,
		d_Queue, d_QueueI, maxQueueSize, VertexNum, 0, 0);
#endif

}

void runPrimHybrid(int iter) {
	if (iter == 1)
        runPrimMin(); 
    else
        runPrimRadixSort(); 
}

void removeParallelEdgesRadix()
{
	if (EdgeNum < REDUNDANT_MIN_E) 
		return ; 

    int VBits = (int)((log((double) VertexNum) / log(2.0))) + 1;
	
    InitializeIndexInvoker(dimGrid, dimBlock, d_Eval, EdgeNum); 

    // Sort by EJ, since it's already sorted by EW and then EI
    cutilSafeCall( cudaMemcpy( d_EJNew, d_EJ, EdgeNum * sizeof(VTYPE), cudaMemcpyDeviceToDevice) );
    cudppSort(planRSort, d_EJNew, (void*)d_Eval, VBits, EdgeNum);

    // Gather the EI of the newly sorted graph
    GatherInvoker(dimGrid, dimBlock, d_EI, d_EINew, d_Eval, EdgeNum);   	
    
    // Mark parallel edges
    unsigned int *d_marker = d_EWNew; 
    Set1Invoker(dimGrid, dimBlock, d_marker, EdgeNum); 
    
    MarkDuplicatedEdgesInvoker(dimGrid, dimBlock, d_EINew, d_EJNew, d_Eval, d_marker, EdgeNum); 

    // Compact
    InitializeIndexInvoker(dimGrid, dimBlock, d_Ekey, EdgeNum); 
    cudppCompact(planCompact, d_Eval, (size_t *) d_NEWEdgeNum, d_Ekey, d_marker, EdgeNum); 
    
    //int oldEdgeNum = EdgeNum; 
    cutilSafeCall( cudaMemcpy(&EdgeNum, d_NEWEdgeNum, sizeof(VTYPE), cudaMemcpyDeviceToHost) );
    //cout << oldEdgeNum << " --> " << EdgeNum << endl;  

    GatherInvoker(dimGrid, dimBlock, d_EI, d_EINew, d_Eval, EdgeNum); 

    swapVTYPE(&d_EI, &d_EINew); 

	#ifdef OUTPUT
		GatherEsInvoker(dimGrid, dimBlock, d_Eval, d_ESIndex, d_EW, d_EWNew, d_EI, 
			d_EJ, d_EJNew, d_EOrg, d_EOrgNew, EdgeNum, VertexNum);
	#else
		GatherEsInvoker(dimGrid, dimBlock, d_Ekey, d_Eval, d_ESIndexNew, d_EW, d_EWNew, d_EI, d_EINew, 
			d_EJ, d_EJNew, 0, 0, EdgeNum, VertexNum);
	#endif

    // Prepare for next iterartion
    swapVTYPE(&d_EW, &d_EWNew); 
    swapVTYPE(&d_EJ, &d_EJNew); 

#ifdef OUTPUT
    swapVTYPE(&d_EOrg, &d_EOrgNew); 
#endif
}


bool mergeComponents() {
	//cout << iter << ": VertexNum = " << VertexNum << " EdgeNum = " << EdgeNum << endl; 

    // VisitList has some value > VertexNum during Prim. Fix it here
	FixingOutPointersInvoker(dimGrid, dimBlock, d_VisitList, VertexNum);

    // Fixing the visit list in a way that each vertex points to its representative directly
	do {		// Repeatedly go back to root
        cutilSafeCall(cudaMemset(d_flag, 0, sizeof(bool)));

        FixingPointersInvoker(512, 128, d_VisitList, d_flag, VertexNum);
    
        cutilSafeCall(cudaMemcpy(flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));
    
    } while (*flag != 0) ;

    // Fixing the IDs of components and bringing the vertices of one component together
    //    Constructing the tags
    ConstructTagInvoker(dimGrid, dimBlock, d_VisitList, d_ESIndex, d_tag, VertexNum);
    //    Exclusive scan of tags to find new ID for each component
    
    //Prefix sum on the tags to reconstruct the components with their new IDs
    cudppScan(planScan, d_tag, d_tag, VertexNum + 1);
	// Output is the same as input, any potential problem?

    // Get the number of components
    cutilSafeCall(cudaMemcpy(&CompNum, &d_tag[VertexNum], sizeof(VTYPE), cudaMemcpyDeviceToHost));

    if (CompNum == 1 || (CompNum == VertexNum))	// Done, only one component left
		return false; 

    // Fixing the Visitlist by changing the ID of each vertex to its new ID using tag
    UpdateVertexIDsInvoker(512, 64, d_tag, d_VisitList, VertexNum);

    // Construct EI, EJ, mark edges for removal
    unsigned int *d_marker = d_EINew; 

	ConstructEIInvoker(512, 64, d_VisitList, d_EI, d_EJ, d_EW, EdgeNum, d_Ekey, d_marker);

    cudppCompact(planCompact, d_Eval, (size_t *) d_NEWEdgeNum, d_Ekey, d_marker, EdgeNum); 
    cutilSafeCall( cudaMemcpy(&EdgeNum, d_NEWEdgeNum, sizeof(VTYPE), cudaMemcpyDeviceToHost) );

    //printf("%i\n", EdgeNum); 

    if (EdgeNum == 0)
        return false; 

    cudaThreadSynchronize();
	cutilCheckError( cutStartTimer(timerTest) );

	if (primAlgo == PRIM_RADIXSORT || primAlgo == PRIM_HYBRID || EdgeNum >= REDUNDANT_MIN_E) {
		// Prepare EKey
		GatherInvoker(dimGrid, dimBlock, d_EW, d_Ekey, d_Eval, EdgeNum); 

		// Sort by edge weight
		cudppSort(planSort, d_Ekey, (void*) d_Eval, 32, EdgeNum);            			 
	}

	cudaThreadSynchronize();
	cutilCheckError( cutStopTimer(timerTest) );

	// Now gather EI
	GatherInvoker(dimGrid, dimBlock, d_EI, d_EINew, d_Eval, EdgeNum);
	swapVTYPE( &d_EI , &d_EINew); 

	// Then sort by starting vertex to bring edges from the same new vertex together
    VertexNum = CompNum; // Update the number of vertices
    int idBits = (int)((log((double) VertexNum) / log(2.0))) + 1;

    cudppSort(planSort, d_EI, (void*) d_Eval, idBits, EdgeNum);            			 

    // Constructing EWN, EJN, ESIN
    cudaMemset(d_ESIndex, M, (VertexNum+1) * sizeof(VTYPE)); 

#ifdef OUTPUT
    GatherEsInvoker(dimGrid, dimBlock, d_Eval, d_ESIndex, d_EW, d_EWNew, d_EI, 
		d_EJ, d_EJNew, d_EOrg, d_EOrgNew, EdgeNum, VertexNum);
#else
    GatherEsInvoker(dimGrid, dimBlock, d_Ekey, d_Eval, d_ESIndexNew, d_EW, d_EWNew, d_EI, d_EINew, 
		d_EJ, d_EJNew, 0, 0, EdgeNum, VertexNum);
#endif

    // Prepare for next iterartion
    swapVTYPE(&d_EW, &d_EWNew); 
    swapVTYPE(&d_EJ, &d_EJNew); 

#ifdef OUTPUT
    swapVTYPE(&d_EOrg, &d_EOrgNew); 
#endif
	
	cudaThreadSynchronize(); 
    cutilCheckError( cutStartTimer(timerParallel) );
  
    //removeParallelEdgesRadix(); 
	
	cudaThreadSynchronize(); 
    cutilCheckError( cutStopTimer(timerParallel) );
	
	return true;		// One more round
}

#ifdef OUTPUT
VTYPE trace(VTYPE v) {
	VTYPE root = v; 

	while (parent[root] != M) 
		root = parent[root]; 
	
	if (v != root)   // Compress the path
		while (parent[v] != root) {
			int next = parent[v]; 
			parent[v] = root; 
			v = next; 
		}

	return root; 
}

void removeDuplicateEdges() {
    mstEdgeNo = 0;
    totalWeight = 0; 

	memset(parent, M, SizeVV); 

    for(VTYPE e = 0; e < EdgeNum; e++) {
        if (EIFlag[e] == 0)
			continue;

		// Last edge of a component, can be duplicated!
        if (EIFlag[e] == 2) {		
            int u = trace(EI[e]);
            int v = trace(EJ[e]); 

            if (u == v)		// Duplicated edge
                continue;       

            parent[u]=v;
        }

        mstEdgeNo++;
        totalWeight += EW[e]; 
    }

    printf("%i %u\n", mstEdgeNo, totalWeight); 
}
#endif

// Using segmented sort
void PMST() 
{
 		cudaThreadSynchronize(); 
		cutilCheckError( cutStartTimer(timerTotal) );

		cutilCheckError( cutStartTimer(timerInit) );
	initializeParameters(); 
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerInit) );

		cutilCheckError( cutStartTimer(timerAlloc) );
	allocateMemory(); 
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerAlloc) ); 

		cutilCheckError( cutStartTimer(timerCopy) );
	copyInput(); 
		cudaThreadSynchronize(); 
		cutilCheckError( cutStartTimer(timerCopy) );

		cutilCheckError( cutStartTimer(timerInit) );
	initializeGPUData(); 
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerInit) );

	iter = 0; 
    bool MergeResult;
	// Multiple iterations, until only one component left
	cutilCheckError( cutStartTimer(timerExec) );

	if (primAlgo == PRIM_RADIXSORT)
		sortEdgeList(); 

    while(true) {        
        if (EdgeNum == 0)
            break; 

        iter++; 
		//cout<<"Iter="<<iter<<" ";
		// Initialize
	    cutilSafeCall( cudaMemset(d_VisitList, M, VertexNum * sizeof(VTYPE)) );
	    cutilSafeCall( cudaMemset(d_Queue, M, QueueSize * sizeof(VTYPE)) );

		cudaThreadSynchronize(); 
		cutilCheckError( cutStartTimer(timerVPrim) );
		cutilCheckError( cutStartTimer(timerIter) );     

		switch (primAlgo) {
			case PRIM_RADIXSORT: 
				runPrimRadixSort(); 
				break; 
			case PRIM_MIN: 
				runPrimMin(); 
				break; 
			case PRIM_HYBRID: 
				runPrimHybrid(iter); 
				break; 
		}
		
		//cout << "Iter " << iter << ": " << cutGetTimerValue(timerParallel) << "ms" << endl; 

		cudaThreadSynchronize();
		cutilCheckError( cutStopTimer(timerVPrim) );

		cutilCheckError( cutStartTimer(timerMerge) );
		MergeResult=mergeComponents();
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerMerge) );

		cudaThreadSynchronize();
		cutilCheckError( cutStopTimer(timerIter) );
   //     cout << "Iter " << iter << ": " << (int) cutGetTimerValue(timerIter) << "ms" 
			//<< ", Prim: " << (int) cutGetTimerValue(timerVPrim) 
			//<< ", Merge " << (int) cutGetTimerValue(timerMerge)
			//<< ", Parallel " << (int) cutGetTimerValue(timerParallel) << endl; 
		cutilCheckError( cutResetTimer(timerIter) );

		if (!MergeResult)		// Only one left of number of components not changed
        {
            break; 
        }
	}

	cudaThreadSynchronize(); 
	cutilCheckError( cutStopTimer(timerExec) );

	// Restore the number of vertices and edges
	VertexNum = orgVertexNum; 
	EdgeNum = orgEdgeNum; 
   
    // Copy the MST back to CPU
#ifdef OUTPUT
		cutilCheckError( cutStartTimer(timerCopy) );
    cutilSafeCall( cudaMemcpy(EIFlag, d_EIFlag, EdgeNum * sizeof(BYTE), cudaMemcpyDeviceToHost) );
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerCopy) );

		cutilCheckError( cutStartTimer(timerDup) );
	removeDuplicateEdges(); 
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerDup) );
#endif

	// DONE!
		cudaThreadSynchronize(); 
		cutilCheckError( cutStopTimer(timerTotal) );

	// Log and report the result
//	reportResult(); 

    // Deallocations
	releaseMemory(); 

    return;
}


/************************************************************************/
/* Main function                                                        */
/************************************************************************/

void addEdge(VTYPE *dad, VTYPE u, VTYPE v, VTYPE &noComp) 
{
    VTYPE du = u, dv = v; 

    while (dad[du] != -1) 
        du = dad[du]; 

    while (dad[dv] != -1) 
        dv = dad[dv]; 

    while (u != du) {
        int tmp = dad[u]; 
        dad[u] = du; 
        u = tmp; 
    }

    while (v != dv) {
        int tmp = dad[v]; 
        dad[v] = dv; 
        v = tmp; 
    }

    if (du != dv) { 
        noComp--; 
        dad[du] = dv; 
    }
}

void make_connected(VTYPE * &ESIndex, VTYPE * &EJ, WTYPE * &EW, VTYPE VertexNum, VTYPE &EdgeNum)
{
    VTYPE *dad = new VTYPE[VertexNum]; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        dad[i] = -1; 

    noComp = VertexNum; 

    int ec = 0; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        for (VTYPE j = ESIndex[i]; j < ESIndex[i+1]; j++) {
            addEdge(dad, i, EJ[j], noComp); 
            ec++; 
        }

    printf("No. Component: %i\n", noComp); 

    int newNoEdge = EdgeNum + (noComp - 1) * 2; 

    VTYPE *ESIndexNew = new VTYPE[newNoEdge]; 
    VTYPE *EJNew = new VTYPE[newNoEdge]; 
    WTYPE *EWNew = new WTYPE[newNoEdge]; 

    VTYPE *addedV = new VTYPE[VertexNum]; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        addedV[i] = 0; 

    int last = -1; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        if (dad[i] == -1)  {
            if (last != -1) 
            {
                addedV[last]++; 
                addedV[i]++; 
            }
            last = i; 
        }

    VTYPE added = 0; 

    for (VTYPE i = 0; i < VertexNum; i++) {
        ESIndexNew[i] = ESIndex[i] + added; 
        added += addedV[i]; 
    }

    ESIndexNew[VertexNum] = newNoEdge; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        for (int j = ESIndex[i]; j < ESIndex[i+1]; j++)  {
            EJNew[ESIndexNew[i] + j - ESIndex[i]] = EJ[j]; 
            EWNew[ESIndexNew[i] + j - ESIndex[i]] = EW[j]; 
        }

    last = -1; 

    WTYPE count = EdgeNum; 

    for (VTYPE i = 0; i < VertexNum; i++) 
        if (dad[i] == -1)  {
            if (last != -1) 
            {
                EJNew[ESIndexNew[i+1]-addedV[i]] = last; 
                EWNew[ESIndexNew[i+1]-addedV[i]] = count; 

                EJNew[ESIndexNew[last+1]-addedV[last]] = i; 
                EWNew[ESIndexNew[last+1]-addedV[last]] = count; 

                addedV[last]--; 
                addedV[i]--; 
                count++; 
            }
            last = i; 
        }

    free(ESIndex); 
    free(EJ); 
    free(EW); 

    ESIndex = ESIndexNew; 
    EJ = EJNew; 
    EW = EWNew; 

    EdgeNum = newNoEdge; 

    SizeEV = sizeof(VTYPE) * (EdgeNum);
    SizeEW = sizeof(WTYPE) * (EdgeNum);
}

int main(int argc, char* argv[])
{
    if(!InitCUDA()) {
        return 0;
    }

    parse_args(argc,argv);

    CUT_DEVICE_INIT(1, argv);

	// Reading input file
	if (!inputBinary) 
		readTextInput(); 
	else
		readBinaryInput(); 

    make_connected(ESIndex, EJ, EW, VertexNum, EdgeNum); 

    srand(50); 

    // Modify the weight by truncating just 8 bits and merge it with destination ID
    /*
	for (int i = 0; i < VertexNum; i++) {
        int start = ESIndex[i]; 
        int end = ESIndex[i+1]; 

        for (int j = start; j < end; j++) 
            if (EJ[j] >= i) { 
                int w = EW[j]; 
                int nw = j;//( (rand()+1) << 24) | j; //w;

                EW[j] = nw; 

                for (int t = ESIndex[EJ[j]]; t < ESIndex[EJ[j]+1]; t++) 
                    if (EJ[t] == i && EW[t] == w) {
                        EW[t] = nw; 
                        break; 
                    }
            }
    }
    
    */

    fname.append(".log"); 
    ofstream Logs("PMSF.LOG",ios_base::app);

	cutilCheckError( cutCreateTimer(&timerBoruvka) );
	cutilCheckError( cutCreateTimer(&timerKruskal) );
	cutilCheckError( cutCreateTimer(&timerPrim) );
	cutilCheckError( cutCreateTimer(&timerGConst) );

    if(runs>0){

		cutilCheckError( cutCreateTimer(&timerTotal) );
		cutilCheckError( cutCreateTimer(&timerInit) );
		cutilCheckError( cutCreateTimer(&timerAlloc) );
		cutilCheckError( cutCreateTimer(&timerCopy) );
		cutilCheckError( cutCreateTimer(&timerExec) );
		cutilCheckError( cutCreateTimer(&timerDup) );
		cutilCheckError( cutCreateTimer(&timerVPrim) );
		cutilCheckError( cutCreateTimer(&timerMerge) );
		cutilCheckError( cutCreateTimer(&timerTest) );
		cutilCheckError( cutCreateTimer(&timerParallel) );
		cutilCheckError( cutCreateTimer(&timerIter) );
	    
		PMST();

		 for (int t = 0; t < 10; t++) 
			stepTime[t] = 0.0; 

		cutilCheckError( cutResetTimer(timerTotal) );
		cutilCheckError( cutResetTimer(timerInit) );
		cutilCheckError( cutResetTimer(timerAlloc) );
		cutilCheckError( cutResetTimer(timerCopy) );
		cutilCheckError( cutResetTimer(timerExec) );
		cutilCheckError( cutResetTimer(timerDup) );
		cutilCheckError( cutResetTimer(timerVPrim) );
		cutilCheckError( cutResetTimer(timerMerge) );
		cutilCheckError( cutResetTimer(timerTest) );
		cutilCheckError( cutResetTimer(timerParallel) );
		int t = 0; 
		while(1)  {
			printf("%i\n", t); 
			PMST();
	        
			t++;
			if (cutGetTimerValue(timerExec) / (t) > 5000.0 || t == runs)
				break; 
		}

		stepTime[0] = cutGetTimerValue(timerTotal) / double(t); 
		stepTime[1] = cutGetTimerValue(timerAlloc) / double(t); 
		stepTime[2] = cutGetTimerValue(timerCopy) / double(t); 
		stepTime[3] = cutGetTimerValue(timerExec) / double(t); 
		stepTime[4] = cutGetTimerValue(timerDup) / double(t); 
		stepTime[5] = cutGetTimerValue(timerVPrim) / double(t); 
		stepTime[6] = cutGetTimerValue(timerMerge) / double(t); 
		stepTime[7] = cutGetTimerValue(timerTest) / double(t); 
		stepTime[8] = cutGetTimerValue(timerParallel) / double(t); 

		// Log to file
		switch (primAlgo) {
			case PRIM_RADIXSORT: 
				cout<<"SortPARMSF ";
				Logs<<"SortPARMSF ";
				break;
			case PRIM_MIN: 
				cout<<"MinPARMSF ";
				Logs<<"MinPARMSF ";
				break; 
			case PRIM_HYBRID: 
				cout<<"HPARMSF ";
				Logs<<"HPARMSF ";
				break; 
		}

		cout << maxQueueSize <<" Iter:"<< iter<<" Test:"<< stepTime[7] <<" Exec:" <<stepTime[3] << " VPrim:" << stepTime[5] << " Merge:" << stepTime[6] << " Parallel:" << stepTime[8] << endl;
		Logs << fname.c_str() << " V:" << orgVertexNum << " E:" << orgEdgeNum << " RUNS:" << t <<" Q:" << maxQueueSize <<" Iter:"<< iter<<" Test:"<< stepTime[7] <<" Exec:" <<stepTime[3] << " VPrim:" << stepTime[5] << " Merge:" << stepTime[6] << " Parallel:" << stepTime[8] ;
		cout<<totalWeight<<endl;

		cutilCheckError( cutDeleteTimer(timerTotal) );
		cutilCheckError( cutDeleteTimer(timerInit) );
		cutilCheckError( cutDeleteTimer(timerAlloc) );
		cutilCheckError( cutDeleteTimer(timerCopy) );
		cutilCheckError( cutDeleteTimer(timerExec) );
		cutilCheckError( cutDeleteTimer(timerDup) );
		cutilCheckError( cutDeleteTimer(timerVPrim) );
		cutilCheckError( cutDeleteTimer(timerMerge) );
		cutilCheckError( cutDeleteTimer(timerTest) );
		cutilCheckError( cutDeleteTimer(timerParallel) );
		cutilCheckError( cutDeleteTimer(timerIter) );

    }
	if(CPU){

    //Original Algorithms
    cutilCheckError( cutResetTimer(timerBoruvka) );
    cutilCheckError( cutResetTimer(timerPrim) );
    cutilCheckError( cutResetTimer(timerKruskal) );
    cutilCheckError( cutResetTimer(timerGConst) );
       
    OriginalMSTs(&Logs); 
    
    cout << " Times Boruvka: " << cutGetTimerValue(timerBoruvka); 
    
    cout << " Construction: " << cutGetTimerValue(timerGConst); 
    cout << " Prim: " << cutGetTimerValue(timerPrim); 
    cout << " Kruskal: " << cutGetTimerValue(timerKruskal) << endl; 
    
    Logs<<" Kruskal "<<cutGetTimerValue(timerKruskal);//+cutGetTimerValue(timerGConst);
    Logs<<" Prim "<<cutGetTimerValue(timerPrim);//+cutGetTimerValue(timerGConst);
    Logs<<" Boruvka "<<cutGetTimerValue(timerBoruvka);
    
    cutilCheckError( cutDeleteTimer(timerBoruvka) );
    cutilCheckError( cutDeleteTimer(timerPrim) );
    cutilCheckError( cutDeleteTimer(timerKruskal) );
    cutilCheckError( cutDeleteTimer(timerGConst) );
	}
	Logs<<endl;
    Logs.close();
    
//	CUT_EXIT(argc, argv);
}


WTYPE Boruvka(unsigned int * iter){
    VTYPE i,index,ComponentNum=VertexNum,ComponentNumOLD=0;
    *iter=0;
    VTYPE c1,c2;
    bool * Eflag;
    
    WTYPE total=0;
    VTYPE * EdgeI,* EdgeS;
    VTYPE * id;

    id=(VTYPE *)malloc(VertexNum * sizeof(VTYPE));
    EdgeI=(VTYPE *)malloc(VertexNum * sizeof(VTYPE));
    EdgeS=(VTYPE *)malloc(VertexNum * sizeof(VTYPE));
    Eflag=(bool *)malloc(EdgeNum * sizeof(bool));
    memset(Eflag,0,EdgeNum* sizeof(bool));
    
    for (i = 0; i < VertexNum; i++) {
        id[i] = i;
    }

    while( ComponentNum != 1 && ComponentNumOLD != ComponentNum ) {
        (*iter)++;
        ComponentNumOLD = ComponentNum;
        memset(EdgeI,M,VertexNum* sizeof(VTYPE));
        
        for (i = 0; i < VertexNum; i++) 
            for (index = ESIndex[i]; index < ESIndex[i+1]; index++) {

                
                c1=i;
                while (c1 != id[c1])
                    c1 = id[c1];
                
                c2=EJ[index];
                while (c2 != id[c2])
                    c2 = id[c2];
                
                if (c1 != c2)
                {
                    if ( EdgeI[c1] >= EdgeNum  || EW[index] < EW[EdgeI[c1]])
                    {
                        EdgeI[c1] = index;
                        EdgeS[c1] = i;
                    }
                    if ( EdgeI[c2] >= EdgeNum || EW[index] < EW[EdgeI[c2]])
                    {
                        EdgeI[c2] = index; 
                        EdgeS[c2] = i;
                    }
                }
            }

            for (i = 0; i <VertexNum; i++)
                if (EdgeI[i] < EdgeNum)
                {
                    c1=EdgeS[i];
                    while (c1 != id[c1])
                        c1 = id[c1];

                    c2=EJ[EdgeI[i]];
                    while (c2 != id[c2])
                        c2 = id[c2];

                    if(c1!=c2)
                    {
                        //unite
                        id[c1] = c2;
                        
                        total += EW[EdgeI[i]];
                        ComponentNum --;
                        Eflag[EdgeI[i]]=1;
                    }
                }
                
    }

    return total;

    free(id); 
    free(EdgeI); 
    free(EdgeS); 
    free(Eflag); 
}




using namespace boost;

typedef adjacency_list < vecS, vecS, undirectedS,
no_property, property < edge_weight_t, int > > Graph;
typedef graph_traits < Graph >::edge_descriptor Edge;
typedef graph_traits < Graph >::vertex_descriptor Vertex;
typedef std::pair<int, int> E;


void OriginalMSTs(ofstream * Logs){
    //Boruvka
    VTYPE iter=0;
    //timer.Start(); 
    cutilCheckError( cutStartTimer(timerBoruvka) );
    
    totalWeight = Boruvka(&iter);
    
    cutilCheckError( cutStopTimer(timerBoruvka) );
    
    //timer.Stop();
    cout << "Boruvka Iterations ="<<iter<<endl;
    cout<<"Weights: Boruvka = " << totalWeight; 

    // Boost Init
    E *edge_array; 
    WTYPE *newEW; 
    edge_array=(E *)malloc((EdgeNum / 2) * sizeof(E));
    newEW=(WTYPE *)malloc((EdgeNum / 2) * sizeof(WTYPE));

    if (edge_array == 0 || newEW == 0) {
        printf("Out of memory!\n"); 
        return ; 
    }

    VTYPE edgeId = 0; 

    try {
        for (VTYPE i = 0; i < VertexNum; i++) 
            for (VTYPE j = ESIndex[i]; j < ESIndex[i+1]; j++) 
                if (i < EJ[j]) {
                    edge_array[edgeId] = E(i, EJ[j]); 
                    newEW[edgeId] = EW[j]; 
                    edgeId++; 
                }


	    //Boost Graph Const.
        
        cutilCheckError( cutStartTimer(timerGConst) );

        Graph g(edge_array, edge_array + edgeId, newEW, VertexNum);

        cutilCheckError( cutStopTimer(timerGConst) );

        //Boost Kruskal

        cutilCheckError( cutStartTimer(timerKruskal) );
        
        vector < Edge > spanning_tree;

        kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

        cutilCheckError( cutStopTimer(timerKruskal) );
        
        vector < graph_traits < Graph >::vertex_descriptor > p(num_vertices(g)); 
        
        //Boost Prim

        cutilCheckError( cutStartTimer(timerPrim) );
        
        prim_minimum_spanning_tree(g, &p[0]);

        cutilCheckError( cutStopTimer(timerPrim) );
        
        WTYPE totalWeight = 0; 
        property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);

        for (vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) 
            totalWeight += weight[*ei];

        cout << " Kruskal = " << totalWeight; 


        totalWeight = 0; 

        for (VTYPE i = 0; i != p.size(); i++) 
            if (p[i] != i) {
                for (VTYPE j = ESIndex[i]; j < ESIndex[i+1]; j++) 
                    if (EJ[j] == p[i]) {
                        totalWeight += EW[j];
                        break; 
                    }    
            }

        cout << " Prim = " << totalWeight << endl; 

    } catch (std::exception &e) {
        printf("Out of memory!\n"); 
        return ; 
    }

    delete [] edge_array; 
    delete [] newEW; 
}
