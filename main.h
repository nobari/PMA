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

#include <time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>


#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include "cutil.h"
#include "cudpp\cudpp.h"

typedef unsigned int			VTYPE;
typedef unsigned int			UINT;
typedef unsigned int			WTYPE;
typedef unsigned char			BYTE; 

const UINT M			= 4294967295;
const int MAX_Threads	= 512;

#define SHARED_SIZE_LIMIT		512

using namespace std;

// CUDA Invokers
void InitializeInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE* EI, VTYPE* ESIndex, VTYPE VertexNum);
void InitializeIndexInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *Index, int Num);
void Set1Invoker(dim3 dimGrid, dim3 dimBlock, VTYPE *Index, int Num);

void VisitingPrimInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, WTYPE *EW,
						 VTYPE *EJ, VTYPE *Queue, VTYPE *QueueI, VTYPE QueueSize, VTYPE VertexNum,
						 BYTE *FlagMST,VTYPE *EOrg);

void VisitingPrimNoSortInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, 
							   WTYPE *EW,VTYPE *EJ, VTYPE *Queue, VTYPE *QueueI, VTYPE QueueSize, 
							   VTYPE VertexNum, BYTE *FlagMST,VTYPE *EOrg);

void FixingOutPointersInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE VertexNum);

void FixingPointersInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, bool *flag, VTYPE VertexNum); 

void GatherEsInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *Eval, VTYPE *ESIndex, 
					 WTYPE *EW, WTYPE *EWNew, VTYPE *EI, 
					 VTYPE *EJ, VTYPE *EJNew, VTYPE *EOrg, VTYPE *EOrgNew, 
					 VTYPE EdgeNum, VTYPE VertexNum); 

void ConstructTagInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, VTYPE *Tag, VTYPE VertexNum);

void ConstructEIInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *d_VisitList, VTYPE* EI, VTYPE * EJ, 
						WTYPE * EW, VTYPE EdgeNum, VTYPE *Eval, VTYPE *marker);

void FindNewEdgeNumInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE EdgeNum, VTYPE * Ekey, VTYPE * NewEdgeNum);
void GatherInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *src, VTYPE *dst, VTYPE *gathers, VTYPE size);
void ConstructNewVisitlistByTagInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE VertexNum, VTYPE *Tag);
void UpdateVertexIDsInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *IDs, VTYPE *data, VTYPE Num); 

void MarkDuplicatedEdgesInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *EI, VTYPE *EJ, VTYPE *keys, unsigned int *marker, int EdgeNum); 
void UpdateESIndexInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *EI, VTYPE *ESIndex, int VertexNum, int EdgeNum); 

// declaration, forward
void readTextInput(); 
void readBinaryInput(); 
void parse_args(int argc, char **argv);
void OriginalMSTs(ofstream * Logs);

