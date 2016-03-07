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

#include "main.h"

texture<uint1> texUInt, texUInt1, texUInt2; 

/************************************************************************/
/* PMST kernels                                                         */
/************************************************************************/
__global__ void Initialize(VTYPE *EI,VTYPE *ESIndex, VTYPE VertexNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < VertexNum; pos += totalThreads)
    {
        //For each vertex
        int StartIndex = ESIndex[pos];
        int EndIndex = ESIndex[pos+1];

		if (EndIndex == M) 
			continue; 

        for (int i = StartIndex; i < EndIndex; i++)
            EI[i] = pos;
    }
}

__global__ void InitializeIndex(VTYPE *data, int Num) {
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for (VTYPE pos = tid; pos < Num; pos += totalThreads)
	{
		data[pos] = pos;
	}
}

__global__ void Set1(VTYPE *data, int Num) {
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for (VTYPE pos = tid; pos < Num; pos += totalThreads)
	{
		data[pos] = 1;
	}
}

extern __shared__ VTYPE s_data[]; 

template<bool output>
__global__ void VisitingPrim(VTYPE *VisitList, VTYPE *ESIndex, WTYPE *EW, VTYPE *EJ, 
							 VTYPE *Queue, VTYPE *QueueI, VTYPE QueueSize, VTYPE VertexNum, 
							 BYTE *FlagMST, VTYPE * EOrg)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

	VTYPE *s_QueueS = Queue + tid * QueueSize; //s_data + (threadIdx.x * 2 + 0) * QueueSize;
	VTYPE *s_QueueE = QueueI + tid * QueueSize; //s_data + (threadIdx.x * 2 + 1) * QueueSize;

    int workBlock = (VertexNum - 1) / totalThreads + 1; 

    int startBlock = workBlock * tid; 
    int endBlock = MIN(startBlock + workBlock, VertexNum); 

    for(VTYPE Root = startBlock; Root < endBlock; Root++)
    {
        // For each vertex
        // Check visited or not
        // Compare the visitlist item to M if it is equal to M then change it to Root, visit it
        // therefore, if the return value, old value, was M it means that vertex Root was unvisited
        VTYPE VisitedBy = atomicCAS(&VisitList[Root], M, Root);

        if (VisitedBy == M)		// Not visited
        {
            // Clear the Queue
            VTYPE QI = 0;

            // Insert Root to Q
            int StartIndex, EndIndex; 

            VTYPE minId;
            WTYPE minW = M;

            s_QueueS[QI] = ESIndex[Root];
            s_QueueE[QI] = ESIndex[Root+1];
			QI++; 

            if (s_QueueS[0] == s_QueueE[0] || s_QueueE[0] == M)       // Island
                continue; 

            //Start
            while (QI < QueueSize)
            {
                minW = M; 

                //Find the minimum outgoing edge from Q
                // foreach T in Q
                for (VTYPE T = 0; T < QI; T++)
                {
                    StartIndex = s_QueueS[T];
                    EndIndex = s_QueueE[T];

					if (StartIndex == M || EndIndex == M) 
						continue; 

                    //find a J that is not in Q
				    VTYPE J = StartIndex; 
                    for (; J < EndIndex; J++)
					    // No need to use atomic since it's OK if it's being visited by another thread
                        if (VisitList[EJ[J]] != Root) 
                            break;

                    if (J < EndIndex && EW[J] < minW)
                    {
                        minW = EW[J]; 
                        minId = J; 
                    }

                    s_QueueS[T] = J;	// to prune or ignore these edges for next time
                }

                if (minW == M)	// No edge has been found, stop
                    break;

                VTYPE minJ = EJ[minId]; 

                // An edge was found
                // Check its J for visitng
                VisitedBy = atomicCAS(&VisitList[minJ], M, Root);

                if (VisitedBy == M)
                {
                    // J is not visited
                    // Insert it to the queue
                    s_QueueS[QI] = ESIndex[minJ];					// Root vertex id
                    s_QueueE[QI] = ESIndex[minJ+1];			// Root edge index

                    if (output)
                        FlagMST[EOrg[minId]] = 1;			// Mark the edge

					QI++; 
                }
                else
                {
                    // J has been visited
					// We cannot set the VisitList of Root to VisitedBy,
					// 'cause the other thread will think this node is in its component
                    if (VisitedBy >= VertexNum)
                        VisitList[Root] = VisitedBy; 
                    else
                        VisitList[Root] = VertexNum + VisitedBy;

					// We can check later to see whether this edge has been added before.
                    if (output)
                        FlagMST[EOrg[minId]] = 2;
					
                    break; // Stop this component
                }
            }
        }
    }
}

template<bool output>
__global__ void VisitingPrimNoSort(VTYPE *VisitList, VTYPE *ESIndex, WTYPE *EW, VTYPE *EJ, 
							 VTYPE *xQueue, VTYPE *xQueueI, VTYPE QueueSize, VTYPE VertexNum, 
							 BYTE *FlagMST, VTYPE * EOrg)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

	VTYPE *s_QueueS = /*s_data + (threadIdx.x * 2 + 0) * QueueSize;*/ xQueue + tid * QueueSize;
	VTYPE *s_QueueE = /*s_data + (threadIdx.x * 2 + 1) * QueueSize;*/ xQueueI + tid * QueueSize;

    int workBlock = (VertexNum - 1) / totalThreads + 1; 

    int startBlock = workBlock * tid; 
    int endBlock = MIN(startBlock + workBlock, VertexNum); 

    for(VTYPE Root = startBlock; Root < endBlock; Root++)
    {
        // For each vertex
        // Check visited or not
        // Compare the visitlist item to M if it is equal to M then change it to Root, visit it
        // therefore, if the return value, old value, was M it means that vertex Root was unvisited
        VTYPE VisitedBy = atomicCAS(&VisitList[Root], M, Root);

        if (VisitedBy == M)		// Not visited
        {
            // Number of items in the queue
            VTYPE QI = 0;

            // Insert Root to Q
            int StartIndex, EndIndex; 

            VTYPE minId, minJ, sMinId;
            WTYPE minW, sMinW;

            s_QueueS[QI] = ESIndex[Root];
            s_QueueE[QI] = ESIndex[Root+1];

			if (s_QueueS[QI] >= s_QueueE[QI] || s_QueueE[QI] == M) 
				continue; 
			QI++; 

            //Start
            while (QI < QueueSize)
            {
                minW = M; 

                //Find the minimum outgoing edge from Q
                // foreach T in Q
                for (VTYPE T = 0; T < QI; T++)
                {
                    StartIndex = s_QueueS[T];

					if (StartIndex == M) 
						continue; 

                    if (T < QI - 1 && VisitList[EJ[StartIndex]] != Root) 
                        sMinW = EW[StartIndex]; 
                    else
                    {
	                    EndIndex = s_QueueE[T];

						if (EndIndex == M) 
							continue; 

						if (T < QI-1) 
							StartIndex++; 

						sMinW = M; 

                        //find a J that is not in Q
                        for (VTYPE id = StartIndex; id < EndIndex; id++)
                        {
                            WTYPE weight = EW[id]; 

                            if (weight < M) {
					            // No need to use atomic since it's OK if it's being visited by another thread
                                if (VisitList[EJ[id]] == Root) 
                                    EW[id] = M; 
                                else
                                {
                                    if (weight < sMinW)
                                    {
                                        sMinW = EW[id]; 
                                        sMinId = id; 
                                    }
                                }
                            }
                        }

                        // Swap
                        if (sMinW != M && sMinId != StartIndex) {
                            VTYPE tmp = EJ[StartIndex]; 
                            EJ[StartIndex] = EJ[sMinId]; 
                            EJ[sMinId] = tmp; 

                            EW[sMinId] = EW[StartIndex]; 
                            EW[StartIndex] = sMinW; 

                            if (output) {
                                tmp = EOrg[StartIndex]; 
                                EOrg[StartIndex] = EOrg[sMinId]; 
                                EOrg[sMinId] = tmp; 
                            }
                        }

						if (sMinW != M)
							s_QueueS[T] = StartIndex; 
						else
							s_QueueS[T] = M; 
                    }

                    if (sMinW < minW) {
                        minW = sMinW; 
                        minId = StartIndex; 
                    }
                }

                if (minW == M)	// No edge has been found, stop
                    break;

                minJ = EJ[minId]; 

                // An edge was found
                // Check its J for visitng
                VisitedBy = atomicCAS(&VisitList[minJ], M, Root);

                if (VisitedBy == M)
                {
                    // J is not visited
                    // Insert it to the queue
                    int start = ESIndex[minJ]; 
                    int end = ESIndex[minJ + 1]; 

                    if (start < end && end != M) {
                        s_QueueS[QI] = ESIndex[minJ];			
                        s_QueueE[QI] = ESIndex[minJ+1];
                    } else 
                        s_QueueS[QI] = M; 

                    if (output)
                        FlagMST[EOrg[minId]] = 1;			// Mark the edge

					QI++; 
                }
                else
                {
                    // J has been visited
					// We cannot set the VisitList of Root to VisitedBy,
					// 'cause the other thread will think this node is in its component
                    if (VisitedBy >= VertexNum)
                        VisitList[Root] = VisitedBy; 
                    else
                        VisitList[Root] = VertexNum + VisitedBy;

					// We can check later to see whether this edge has been added before.
                    if (output)
                        FlagMST[EOrg[minId]] = 2;
					
                    break; // Stop this component
                }
            }
        }
    }
}

__global__ void FixingOutPointers(VTYPE *VisitList, VTYPE VertexNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < VertexNum; pos += totalThreads)
    {
        VTYPE Father = VisitList[pos];

        if (Father >= VertexNum)
            VisitList[pos] = Father - VertexNum;
    }
}

__global__ void FixingPointers(VTYPE *VisitList, bool *flag, VTYPE VertexNum)
{
	__shared__ bool s_flag;

    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

	if (threadIdx.x == 0) 
		s_flag = false; 

	__syncthreads(); 

    for (VTYPE pos = tid; pos < VertexNum; pos += totalThreads)
    {
        VTYPE Father = VisitList[pos] ; 

		if (pos != Father) {
	        VTYPE GrandFather = VisitList[Father]; 

            if (pos == GrandFather) {		// two-vertex cycle, choose the bigger one as the root
                if(pos<Father)
                    VisitList[pos] = pos ; 

                s_flag = true; 
            } 
			else if (Father != GrandFather) {
                VisitList[pos] = GrandFather;
                s_flag = true;                 
            }
		}
    }

	__syncthreads(); 

	if (threadIdx.x == 0 && s_flag) 
		*flag = true; 
}

__global__ void ConstructTag(VTYPE *VisitList, VTYPE *ESIndex, VTYPE *Tag, VTYPE VertexNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads=gridDim.x * blockDim.x;

    for (VTYPE pos = tid; pos < VertexNum; pos += totalThreads)
		if (VisitList[pos] == pos && ESIndex[pos] != ESIndex[pos+1] && ESIndex[pos] != M && ESIndex[pos+1] != M)
			Tag[pos] = 1; 
		else
			Tag[pos] = 0;

	if (tid == totalThreads)
		Tag[VertexNum+1] = 0; 
}


__global__ void ConstructNewVisitlistByTag(VTYPE *VisitList, VTYPE VertexNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;
    
    for (VTYPE pos = tid; pos < VertexNum; pos += totalThreads)
    {
        // New node ID
        VisitList[pos] = tex1Dfetch(texUInt, VisitList[pos]).x;
    }
}

__global__ void UpdateVertexIDs(VTYPE *data, VTYPE Num)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;
    
    for (VTYPE pos = tid; pos < Num; pos += totalThreads)
        data[pos] = tex1Dfetch(texUInt, data[pos]).x;
}

__global__ void ConstructEI(VTYPE *EI, VTYPE *EJ, WTYPE *EW, VTYPE EdgeNum, VTYPE *Eval, VTYPE *marker)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;
    
    for (VTYPE i = tid; i < EdgeNum; i += totalThreads)
    {
		int u = tex1Dfetch(texUInt, EI[i]).x; 
		int v = tex1Dfetch(texUInt, EJ[i]).x; 

		// pruned, or Same component so prune this edge
		if (u == v)	
			marker[i] = 0; 
		else
		{
			marker[i] = 1; 
			EI[i] = u; 
			EJ[i] = v; 
		}

		Eval[i] = i; 
    }
}

__global__ void FindNewEdgeNumBlock(VTYPE EdgeNum, VTYPE *Ekey, int blockSize, VTYPE *blockId) 
{
    const UINT tid = blockIdx.x * blockDim.x + threadIdx.x;

    VTYPE firstId = tid * blockSize; 
    VTYPE lastId = firstId + blockSize; 
    VTYPE first, last; 

    if (firstId < EdgeNum)
        first = Ekey[firstId]; 
    else
        first = M; 

    if (lastId < EdgeNum)
        last = Ekey[lastId]; 
    else
        last = M; 

    if ((first != M || tid == 0) && last == M) 
        *blockId = tid; 
}

__global__ void FindNewEdgeNum(VTYPE EdgeNum, VTYPE *Ekey, VTYPE *NewEdgeNum)
{
    const UINT tid = threadIdx.x;

    int blockId = *NewEdgeNum; 
    int id = blockId * blockDim.x + tid; 

    __syncthreads(); 

    if (id < EdgeNum - 1 && Ekey[id] != M && Ekey[id+1] == M)
        *NewEdgeNum = id+1; 

    if (id == 0 && Ekey[id] == M) 
        *NewEdgeNum = 0; 
}

template<bool output, bool Index>
__global__ void GatherEs(VTYPE *Eval, VTYPE *ESIndex,
						 WTYPE *EW, WTYPE *EWNew, VTYPE *EI, 
						 VTYPE *EJ, VTYPE *EJNew, VTYPE *EOrg, VTYPE *EOrgNew, 
						 VTYPE EdgeNum, VTYPE VertexNum) 
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < EdgeNum; pos += totalThreads)
    {
        VTYPE id = Eval[pos];

		EJNew[pos]   = tex1Dfetch(texUInt1, id).x;	// EJ[id]
		EWNew[pos]   = tex1Dfetch(texUInt, id).x;	//EW[id]; 

        if (output) 
            EOrgNew[pos] = tex1Dfetch(texUInt2, id).x; //EOrg[id]; 

		if (Index) 
			if (pos==0)
			{
				ESIndex[0] = 0;
				ESIndex[EI[EdgeNum-1]+1] = EdgeNum;

				if (EI[0] != 0) 
					ESIndex[EI[0]] = 0; 
			}
			else if (EI[pos] != EI[pos-1]) {
				ESIndex[EI[pos-1]+1] = pos; 
				ESIndex[EI[pos]] = pos;
			}
    }
}      

__global__ void Gather(VTYPE *src, VTYPE *dst, VTYPE *gathers, VTYPE size)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < size; pos += totalThreads)
    {
        VTYPE id = gathers[pos];
		dst[pos]	 = src[id]; 
    }
}     

__global__ void UpdateESIndex(VTYPE *EI, VTYPE *ESIndex, VTYPE VertexNum, VTYPE EdgeNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < EdgeNum; pos += totalThreads)
        if (pos == 0) {
            ESIndex[0] = 0; 
			ESIndex[EI[EdgeNum-1]+1] = EdgeNum; 

            if (EI[0] != 0) 
                ESIndex[EI[0]] = 0; 
        } else 
			if (EI[pos-1] != EI[pos]) { 
				ESIndex[EI[pos-1]+1] = pos; 
                ESIndex[EI[pos]] = pos; 
			}
}

__global__ void MarkDuplicatedEdges(VTYPE *EI, VTYPE *EJ, VTYPE *keys, unsigned int *marker, VTYPE EdgeNum)
{
    const UINT tid = threadIdx.x + blockIdx.x * blockDim.x;
    const UINT totalThreads = gridDim.x * blockDim.x;

    for(VTYPE pos = tid; pos < EdgeNum; pos += totalThreads)
        if (pos > 0 && EI[pos-1] == EI[pos] && EJ[pos-1] == EJ[pos]) 
            marker[keys[pos]] = 0; 
}

/************************************************************************/
/* Kernel invokers                                                      */
/************************************************************************/
void InitializeInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *EI, VTYPE *ESIndex, VTYPE VertexNum) {
    Initialize<<< dimGrid, dimBlock >>>(EI, ESIndex, VertexNum);
}

void InitializeIndexInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *data, int size) {
    InitializeIndex<<< dimGrid, dimBlock >>>(data, size); 
}

void Set1Invoker(dim3 dimGrid, dim3 dimBlock, VTYPE *data, int size) {
    Set1<<< dimGrid, dimBlock >>>(data, size); 
}

void VisitingPrimInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, 
						 WTYPE *EW, VTYPE *EJ, VTYPE *Queue, VTYPE *QueueI, VTYPE QueueSize,
						 VTYPE VertexNum, BYTE *FlagMST, VTYPE *EOrg) {
	int shared_mem = 0;//dimBlock.x * QueueSize * sizeof(int) * 2; 

    if (EOrg != 0)
        VisitingPrim<true><<< dimGrid, dimBlock, shared_mem >>>(VisitList, ESIndex, EW, EJ, Queue, QueueI,
    		QueueSize, VertexNum, FlagMST, EOrg);
    else
        VisitingPrim<false><<< dimGrid, dimBlock, shared_mem >>>(VisitList, ESIndex, EW, EJ, Queue, QueueI,
    		QueueSize, VertexNum, FlagMST, EOrg);
}

void VisitingPrimNoSortInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, 
						 WTYPE *EW, VTYPE *EJ, VTYPE *Queue, VTYPE *QueueI, VTYPE QueueSize,
						 VTYPE VertexNum, BYTE *FlagMST, VTYPE *EOrg) {
	int shared_mem = 0;//dimBlock.x * QueueSize * sizeof(int) * 2; 

    if (EOrg != 0)
        VisitingPrimNoSort<true><<< dimGrid, dimBlock, shared_mem >>>(VisitList, ESIndex, EW, EJ, Queue, QueueI,
    		QueueSize, VertexNum, FlagMST, EOrg);
    else
        VisitingPrimNoSort<false><<< dimGrid, dimBlock, shared_mem >>>(VisitList, ESIndex, EW, EJ, Queue, QueueI,
    		QueueSize, VertexNum, FlagMST, EOrg);
}

void FixingOutPointersInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE VertexNum) {
    FixingOutPointers<<< dimGrid, dimBlock >>>(VisitList, VertexNum);
}

void FixingPointersInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, bool *flag, VTYPE VertexNum) {
    FixingPointers<<< dimGrid, dimBlock >>>(VisitList, flag, VertexNum);
}

void ConstructTagInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE *ESIndex, VTYPE *Tag, VTYPE VertexNum) {
    ConstructTag<<< dimGrid, dimBlock >>>(VisitList, ESIndex, Tag, VertexNum);
}

void ConstructNewVisitlistByTagInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE VertexNum, VTYPE *Tag) {
	cudaBindTexture(0, texUInt, Tag); 

	ConstructNewVisitlistByTag<<< dimGrid, dimBlock >>>(VisitList, VertexNum) ;

	cudaUnbindTexture(texUInt); 
}

void UpdateVertexIDsInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *IDs, VTYPE *data, VTYPE Num) 
{
	cudaBindTexture(0, texUInt, IDs); 

	UpdateVertexIDs<<< dimGrid, dimBlock >>>(data, Num); 

	cudaUnbindTexture(texUInt); 
}

void ConstructEIInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *VisitList, VTYPE* EI, VTYPE * EJ, 
						WTYPE *EW, VTYPE EdgeNum, VTYPE *Eval, VTYPE *marker) {
	cudaBindTexture(0, texUInt, VisitList); 

	ConstructEI<<< dimGrid, dimBlock >>>(EI, EJ, EW, EdgeNum, Eval, marker);

	cudaUnbindTexture(texUInt); 
}
 
void FindNewEdgeNumInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE EdgeNum, VTYPE *Ekey, VTYPE *NewEdgeNum) {
    dim3 sBlock = dim3(32); 
    dim3 sGrid = dim3((EdgeNum - 1) / sBlock.x / dimBlock.x + 1); 

    FindNewEdgeNumBlock<<< sGrid, sBlock >>>(EdgeNum, Ekey, dimBlock.x, NewEdgeNum);
    FindNewEdgeNum<<< 1, dimBlock >>>(EdgeNum, Ekey, NewEdgeNum);
}

void GatherEsInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *Eval, VTYPE *ESIndex,
					 WTYPE *EW, WTYPE *EWNew, VTYPE *EI, 
					 VTYPE *EJ, VTYPE *EJNew, VTYPE *EOrg, VTYPE *EOrgNew, 
					 VTYPE EdgeNum, VTYPE VertexNum)  {
	cudaBindTexture(0, texUInt, EW); 
	cudaBindTexture(0, texUInt1, EJ); 

	if (EOrg != 0) {
		cudaBindTexture(0, texUInt2, EOrg); 
		if (ESIndex != 0) 
			GatherEs<true,true><<< dimGrid, dimBlock >>>(Eval, ESIndex, EW, EWNew, EI, EJ, EJNew, 
    			EOrg, EOrgNew, EdgeNum, VertexNum);
		else
			GatherEs<true,false><<< dimGrid, dimBlock >>>(Eval, ESIndex, EW, EWNew, EI, EJ, EJNew, 
    			EOrg, EOrgNew, EdgeNum, VertexNum);
		cudaUnbindTexture(texUInt2); 
	}
    else
		if (ESIndex != 0) 
	        GatherEs<false,true><<< dimGrid, dimBlock >>>(Eval, ESIndex, EW, EWNew, EI, EJ, EJNew, 
				EOrg, EOrgNew, EdgeNum, VertexNum);
		else
			GatherEs<false,false><<< dimGrid, dimBlock >>>(Eval, ESIndex, EW, EWNew, EI, EJ, EJNew, 
    			EOrg, EOrgNew, EdgeNum, VertexNum);

	cudaUnbindTexture(texUInt); 
	cudaUnbindTexture(texUInt1); 
}

void GatherInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *src, VTYPE *dst, VTYPE *gathers, VTYPE size)  {
    Gather<<< dimGrid, dimBlock >>>( src, dst, gathers, size);
}

void MarkDuplicatedEdgesInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *EI, VTYPE *EJ, VTYPE *keys, unsigned int *marker, int EdgeNum) {
    MarkDuplicatedEdges<<< dimGrid, dimBlock >>>(EI, EJ, keys, marker, EdgeNum); 
}

void UpdateESIndexInvoker(dim3 dimGrid, dim3 dimBlock, VTYPE *EI, VTYPE *ESIndex, int VertexNum, int EdgeNum) {
    UpdateESIndex<<< dimGrid, dimBlock >>>(EI, ESIndex, VertexNum, EdgeNum); 
}

