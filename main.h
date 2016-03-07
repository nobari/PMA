/********************************************************************
*  Sadegh Nobari
*  PMA
*********************************************************************/

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

