typedef Node* NodePtr;

// **************************************************
// This kernel performs a direct N^2 collision check
// It is used to find encounters between test particles, which
// are not tested in the gravity pre-checker. 
// For large N, a tree code should be used instead
//
// Date: December 2022
// Author: Simon Grimm
// **************************************************
void collisioncheck_cpu(double4 *x4_h, double *rcritv_h, int *index_h, int *Nencpairs2_h, int2 *Encpairs2_h, const int N1, const int N, const int iy){

	int idx = 0 * 1 + 0 + N1;
	int idy = (0 + iy) * 1 + 0 + N1;

	for(idx = 0 * 1 + 0 + N1; idx < N; ++idx){
		 for(idy = (0 + iy) * 1 + 0 + N1; idy < N; ++idy){
			double4 x4i = x4_h[idx];
			double4 x4j = x4_h[idy];
			double rcritvi = def_pc * rcritv_h[idx];
			double rcritvj = def_pc * rcritv_h[idy];

			bool overlap = true;
			if(x4i.x - rcritvi > x4j.x + rcritvj || x4i.x + rcritvi < x4j.x - rcritvj){
				overlap = false;
			}
			if(x4i.y - rcritvi > x4j.y + rcritvj || x4i.y + rcritvi < x4j.y - rcritvj){
				overlap = false;
			}
			if(x4i.z - rcritvi > x4j.z + rcritvj || x4i.z + rcritvi < x4j.z - rcritvj){
				overlap = false;
			}
			if(idx >= idy){
				overlap = false;
			}
			//ingnore encounters within the same particle cloud
			if(index_h[idx] / WriteEncountersCloudSize_c[0] == index_h[idy] / WriteEncountersCloudSize_c[0]){
				overlap = false;
			}

			if(overlap){
#if def_CPU == 0
				int ne = atomicAdd(&Nencpairs2_h[0], 1);
#else
				int ne;
				#pragma omp atomic capture
				ne = Nencpairs2_h[0]++;
#endif
				Encpairs2_h[ne].x = idx;
				Encpairs2_h[ne].y = idy;
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x - rcriti, x4i.y - rcriti, x4i.z, x4j.x - rcritj, x4j.y - rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x + rcriti, x4i.y - rcriti, x4i.z, x4j.x + rcritj, x4j.y - rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x - rcriti, x4i.y + rcriti, x4i.z, x4j.x - rcritj, x4j.y + rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x + rcriti, x4i.y + rcriti, x4i.z, x4j.x + rcritj, x4j.y + rcritj, x4j.z);
			}
		}
	}
}

// *****************************************************
// This function prints the binary representation of a variable
// Usefull for testing
// *****************************************************
void CheckBinary(unsigned int n){

	printf("%015u ", n);
	for(unsigned int i = 1u << 31; i > 0u; i = i >> 1){
		if((n & i)) printf("1");
		else printf("0");
	}
	printf("\n");
}


// **************************************************
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
// See: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// **************************************************
unsigned int expandBits(unsigned int v){

	v = (v * 0x00010001u) & 0xFF0000FFu;    //65537  0000000000000010000000000000001 & 11111111000000000000000011111111
	v = (v * 0x00000101u) & 0x0F00F00Fu;    //257    0000000000000000000000100000001 & 00001111000000001111000000001111
	v = (v * 0x00000011u) & 0xC30C30C3u;    //17     0000000000000000000000000010001 & 11000011000011000011000011000011
	v = (v * 0x00000005u) & 0x49249249u;    //5      1001001001001001001001001001001 & 01001001001001001001001001001001
	return v;
}

// **************************************************
// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
// See: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// **************************************************
unsigned int morton3D(double4 x4i){
	//Normalize the coordinates to the range 0 to 1
	float xf = (float)((x4i.x + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));
	float yf = (float)((x4i.y + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));
	float zf = (float)((x4i.z + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));

	xf = min(max(xf * 1024.0f, 0.0f), 1023.0f);
	yf = min(max(yf * 1024.0f, 0.0f), 1023.0f);
	zf = min(max(zf * 1024.0f, 0.0f), 1023.0f);

	unsigned int xx = expandBits((unsigned int)xf);
	unsigned int yy = expandBits((unsigned int)yf);
	unsigned int zz = expandBits((unsigned int)zf);

	return xx * 4 + yy * 2 + zz;
}



// *******************************************************
// This funtion reads the last 4 bits of the array morton_h and sorts them with a counting sort routine
// Each threads has to perform a prefix scan on 16 buckets
// This kernel could still be parallelized further
// 256 * 16 is the maximum to be stored in shared memory
//
// Date: October 2022
// Author: Simon Grimm
// *******************************************************


// ***********************************************
// This kernel combines the buckets from the previous thread blocks.
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************

// ***********************************************
// Last part of the radix sort
// This functions calculates the new address in the sorted array 
// The sortIndex_h array contains now the sorted indices of the particle arrays
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************

// ***********************************************
// The array sortRank contains a copy of morton_h.
// morton_h gets sorted now
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************


// *********************************************
// partition part for the quick sort algorithm
// Sortrank can be useda as a temporary copy of morton_h
// https://beginnersbook.com/2015/02/quicksort-program-in-c/
// **********************************************
__host__ void quickSort(unsigned int *morton_h, int2 *sortIndex_h, int first, int last){

	int i, j, pivot;
	unsigned int temp;
	int temp2;

	if(first < last){
		pivot = first;
		i = first;
		j = last;

		while(i < j){
			while(morton_h[i] <= morton_h[pivot] && i < last)
				i++;
			while(morton_h[j] > morton_h[pivot])
				j--;
			if(i < j){
				temp = morton_h[i];
				temp2 = sortIndex_h[i].x;
				morton_h[i] = morton_h[j];
				morton_h[j] = temp;
				sortIndex_h[i].x = sortIndex_h[j].x;
				sortIndex_h[j].x = temp2;
			}

		}


		temp = morton_h[pivot];
		temp2 = sortIndex_h[pivot].x;
		morton_h[pivot] = morton_h[j];
		morton_h[j] = temp;
		sortIndex_h[pivot].x = sortIndex_h[j].x;
		sortIndex_h[j].x = temp2;


		quickSort(morton_h, sortIndex_h, first, j-1);
		quickSort(morton_h, sortIndex_h, j+1, last);
	}

}


void sortCheck_cpu(unsigned int *morton_h, int2 *sortIndex_h, const int N){

	for(int i = 0; i < N; ++i){
		int iid = sortIndex_h[i].x;
		unsigned int a = morton_h[iid];
		printf("%02d ",i);
		CheckBinary(a);
	}
}

void sortCheck2_cpu(unsigned int *morton_h, const int N){

	for(int i = 0; i < N; ++i){
		unsigned int a = morton_h[i];
		printf("%02d ",i);
		CheckBinary(a);
	}
}



void setLeafNode_cpu(Node *leafNodes_h, int2 *sortIndex_h, double4 *x4_h, double *rcritv_h, const int N){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		int ii = sortIndex_h[id].x;
		leafNodes_h[id].isLeaf = true;
		leafNodes_h[id].nodeID = ii;
		leafNodes_h[id].childL = nullptr;
		leafNodes_h[id].childR = nullptr;
		leafNodes_h[id].parent = nullptr;

		leafNodes_h[id].rangeL = id;
		leafNodes_h[id].rangeR = id;

		leafNodes_h[id].counter = 1;	//all threads compute ranges

		double4 x4 = x4_h[ii];
		double rcritv = rcritv_h[ii];

		leafNodes_h[id].xmin = float(x4.x - rcritv);
		leafNodes_h[id].xmax = float(x4.x + rcritv);
		leafNodes_h[id].ymin = float(x4.y - rcritv);
		leafNodes_h[id].ymax = float(x4.y + rcritv);
		leafNodes_h[id].zmin = float(x4.z - rcritv);
		leafNodes_h[id].zmax = float(x4.z + rcritv);


//if(ii == 127 || ii == 493){
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x, x4.y);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x - rcritv, x4.y - rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x - rcritv, x4.y + rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x + rcritv, x4.y - rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x + rcritv, x4.y + rcritv);
//}
	}
}

void setInternalNode_cpu(Node *internalNodes_h, const int N){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		internalNodes_h[id].isLeaf = false;
		internalNodes_h[id].nodeID = id;
		internalNodes_h[id].childL = nullptr;
		internalNodes_h[id].childR = nullptr;
		internalNodes_h[id].parent = nullptr;

		internalNodes_h[id].rangeL = -1;
		internalNodes_h[id].rangeR = -1;

		internalNodes_h[id].counter = 0;	//only 1 thread per node computes parent node

		double r = 10.0 * RcutSun_c[0];

		internalNodes_h[id].xmin =  r;
		internalNodes_h[id].xmax = -r;
		internalNodes_h[id].ymin =  r;
		internalNodes_h[id].ymax = -r;
		internalNodes_h[id].zmin =  r;
		internalNodes_h[id].zmax = -r;
	}
}

// ***********************************************************
// This function computes the highest differing bit as described in 
// Apetri 2014 "Fast and Simple Agglomerative LBVH Construction"
// When the two morton codes are identical, then use the index instead
// see Karras 2012 "Maximizing Parallelism in the Construction of BVHs,
// Octrees, and k-d Tree"
// ***********************************************************
int highestBit(unsigned int *morton_h, int i){

	unsigned int mi = morton_h[i];
	unsigned int mj = morton_h[i + 1];

	if(mi != mj){
		return (mi ^ mj);
	}
	else{
		return (i ^ (i+1)) + 32;
	}
}

// ***********************************************************
// This kernel build a BVH tree
// It uses a bottom up approach as described in Apetri 2014 "Fast and Simple Agglomerative LBVH Construction"
//
// (Apetri 2014: Optional kernel to store delta(i,j) beforehand)
// Date: October 2022
// Author: Simon Grimm
// ***********************************************************
void buildBVH_cpu(unsigned int *morton_h, Node *leafNodes_h, Node *internalNodes_h, const int N){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		int p = 0;
		float xmin;
		float xmax;
		float ymin;
		float ymax;
		float zmin;
		float zmax;

		Node *node = &leafNodes_h[id];

		Node *childL;
		Node *childR;
		Node *parent;

		int rangeL, rangeR;

		for(int i = 0; i < N; ++i){
			if(p >= 0){
#if def_CPU == 0
				if(atomicAdd(&(node->counter), 1) == 1){
#else
				//not parallel yet
				if(node->counter++ == 1){
#endif
					rangeL = node->rangeL;
					rangeR = node->rangeR;

					if(rangeL == 0 || (rangeR != (N - 1) && highestBit(morton_h, rangeR) < highestBit(morton_h, rangeL - 1))){
						parent = &internalNodes_h[rangeR];
						parent->childL = node;
						parent->rangeL = rangeL;
						node->parent = parent;
					}
					else{
						parent = &internalNodes_h[rangeL - 1];
						parent->childR = node;
						parent->rangeR = rangeR;
						node->parent = parent;
					}
//printf("node %d | %d | %u %d %d | %u %d %u\n", i, id, node->nodeID, rangeL, rangeR, parent->nodeID, parent->isLeaf, node->parent->nodeID);

					if(!(node->isLeaf)){
						childL = node->childL;
						childR = node->childR;

						xmin = fminf(childL->xmin, childR->xmin);
						xmax = fmaxf(childL->xmax, childR->xmax);
						ymin = fminf(childL->ymin, childR->ymin);
						ymax = fmaxf(childL->ymax, childR->ymax);
						zmin = fminf(childL->zmin, childR->zmin);
						zmax = fmaxf(childL->zmax, childR->zmax);

						node->xmin = xmin;
						node->xmax = xmax;
						node->ymin = ymin;
						node->ymax = ymax;
						node->zmin = zmin;
						node->zmax = zmax;

//printf("volume %d %d %d %d\n", i, node->nodeID, childL->nodeID, childR->nodeID);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmin, ymin);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmin, ymax);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmax, ymin);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmax, ymax);
					}
					//node is root node
					if(rangeL == 0 && rangeR == N -1){
//printf("root %d %d\n", id, node->nodeID);
						internalNodes_h[N - 1].childR = node; // escape node
						break;
					}

					node = parent;
					//make sure that global memory updates is visible to other threads
					}
					else{
						p = -1;
					break;
				}
			}
		}
	}
}

// ***********************************************************
// This functions checks if two bounding boxes overlap
// ***********************************************************
bool checkOverlap(Node *nodeA, Node *nodeB){

	if(nodeA->xmin > nodeB->xmax || nodeA->xmax < nodeB->xmin){
		return false;
	}
	if(nodeA->ymin > nodeB->ymax || nodeA->ymax < nodeB->ymin){
		return false;
	}
	if(nodeA->zmin > nodeB->zmax || nodeA->zmax < nodeB->zmin){
		return false;
	}
	return true;
}



// ***********************************************************
// This kernel walks down the BVH tree and compares the nodes to a leaf node
// Ever leaf node is processed in parallel
// The kernel uses a local stack to store the traversal path
// See https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
// ***********************************************************
void traverseBVH_cpu(Node *leafNodes_h, Node *internalNodes_h, int *index_h, int *Nencpairs2_h, int2 *Encpairs2_h, unsigned int N1, const int N){
	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		NodePtr stack[64];
		NodePtr *stackPtr = stack;
		*stackPtr++ = nullptr;

		Node *node = internalNodes_h[N - 1].childL;	//root
		Node *leaf = &leafNodes_h[id];

		Node *childL;
		Node *childR;

		for(int i = 0; i < N; ++i){
			//if(id == 520) printf("%d %d\n", i, node->nodeID);
			childL = node->childL;
			childR = node->childR;

			bool overlapL = checkOverlap(leaf, childL);
			bool overlapR = checkOverlap(leaf, childR);

			//remove collisions with j >= i
			if(childL->rangeR <= leaf->rangeR){
				overlapL = false;
			}
			if(childR->rangeR <= leaf->rangeR){
				overlapR = false;
			}
//printf("traverse %d leaf: %d, node: %d L: %d R: %d | %d %d\n", id, leaf->nodeID, node->nodeID, childL->nodeID, childR->nodeID, overlapL, overlapR);

			if(overlapL && childL->isLeaf){
				if(leaf->nodeID >= N1 && childL->nodeID >= N1){

					//ingnore encounters within the same particle cloud
					if(index_h[leaf->nodeID] / WriteEncountersCloudSize_c[0] != index_h[childL->nodeID] / WriteEncountersCloudSize_c[0]){
#if def_CPU == 0
						int ne = atomicAdd(&Nencpairs2_h[0], 1);
#else
						int ne;
						#pragma omp atomic capture
						ne = Nencpairs2_h[0]++;

#endif
						Encpairs2_h[ne].x = leaf->nodeID;
						Encpairs2_h[ne].y = childL->nodeID;
					}
//if(leaf->nodeID == 127 || leaf->nodeID == 493) printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmin, leaf->ymin, leaf->zmin, childL->xmin, childL->ymin, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmin, leaf->ymax, leaf->zmin, childL->xmin, childL->ymax, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmax, leaf->ymin, leaf->zmin, childL->xmax, childL->ymin, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmax, leaf->ymax, leaf->zmin, childL->xmax, childL->ymax, childL->zmin);
				}
			}
			if(overlapR && childR->isLeaf){
				if(leaf->nodeID >= N1 && childR->nodeID >= N1){
					//ingnore encounters within the same particle cloud
					if(index_h[leaf->nodeID] / WriteEncountersCloudSize_c[0] != index_h[childR->nodeID] / WriteEncountersCloudSize_c[0]){
#if def_CPU == 0
						int ne = atomicAdd(&Nencpairs2_h[0], 1);
#else
						int ne;
						#pragma omp atomic capture
						ne = Nencpairs2_h[0]++;
#endif
						Encpairs2_h[ne].x = leaf->nodeID;
						Encpairs2_h[ne].y = childR->nodeID;
					}
//if(leaf->nodeID == 127 || leaf->nodeID == 493) printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmin, leaf->ymin, leaf->zmin, childR->xmin, childR->ymin, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmin, leaf->ymax, leaf->zmin, childR->xmin, childR->ymax, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmax, leaf->ymin, leaf->zmin, childR->xmax, childR->ymin, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmax, leaf->ymax, leaf->zmin, childR->xmax, childR->ymax, childR->zmin);
				}
			}

			//overlap with internal node
			bool traverseL = overlapL && !(childL->isLeaf);
			bool traverseR = overlapR && !(childR->isLeaf);

			if(!traverseL && !traverseR){
				node = *--stackPtr;
			}
			else{
				node = (traverseL) ? childL : childR;
			if(traverseL && traverseR){
				*stackPtr++ = childR;
			}

			}
			if(node == nullptr){
//printf("%d id %d %d reached the end\n", i, id, leaf->nodeID);
			break;
			}
		}
	}
}



void checkNodes_cpu(Node *internalNodes_h, const int N){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N - 1; ++id){
		//printf("internal node %d L %d %u R %d %u | P %d\n", id, internalNodes_h[id].childL->isLeaf, internalNodes_h[id].childL->nodeID, internalNodes_h[id].childR->isLeaf, internalNodes_h[id].childR->nodeID, 0);
		printf("internal node %d L %d %u R %d %u | P %u\n", id, internalNodes_h[id].childL->isLeaf, internalNodes_h[id].childL->nodeID, internalNodes_h[id].childR->isLeaf, internalNodes_h[id].childR->nodeID, internalNodes_h[id].parent->nodeID);

	}
}


__host__ void Data::BVHCall1(){
	int N = N_h[0] + Nsmall_h[0];

#if def_CPU == 0
	for(int i = 0; i < N; i += 32768){
		int Ny = min(N - i, 32768);
		if(Ny > 0){
			collisioncheck_cpu /* dim3((N + 255) / 256, Ny, 1), dim3(256, 1, 1)*/ (x4_h, rcritv_h, index_h, Nencpairs2_h, Encpairs2_h, N_h[0], N, i);
		}
	}
#else
	collisioncheck_cpu /* dim3((N + 255) / 256, Ny, 1), dim3(256, 1, 1)*/ (x4_h, rcritv_h, index_h, Nencpairs2_h, Encpairs2_h, N_h[0], N, 0);

#endif
}

__host__ void Data::BVHCall2(){
	int N = N_h[0] + Nsmall_h[0];

#if def_CPU == 0
	for(unsigned int b = 0; b < 32; b += 4){
		//printf("******** %u *********\n", b);
		sort_kernel <256> <<< (N + 255) / 256, 256 >>> (x4_h, morton_h, sortIndex_h, sortCount_h, sortRank_h, b, N);
		int NN = (N + 255) / 256;
		//printf("NN %d\n", NN); 

		sortmerge_kernel <256> <<< 1, 256 >>> (sortCount_h, NN);
		sortscatter_kernel <<< (N + 255) / 256, 256 >>> (morton_h, sortIndex_h, sortCount_h, sortRank_h, b, N, NN);

//		sortCheck_kernel(morton_h, sortIndex_h, min(N,100));
	}
	sortscatter2_kernel <<< (N + 255) / 256, 256 >>> (morton_h, sortRank_h, sortIndex_h, N);
#else
	for(int i = 0; i < N; ++i){
		sortIndex_h[i].x = i;
		morton_h[i] = morton3D(x4_h[i]);
	}
	//sortCheck2_cpu /* 1, 1 */ (morton_h, N);

	quickSort(morton_h, sortIndex_h, 0, N-1);

#endif

	//sortCheck2_cpu /* 1, 1 */ (morton_h, N);

	setLeafNode_cpu /* (N + 255) / 256, 256 */ (leafNodes_h, sortIndex_h, x4_h, rcritv_h, N);
	setInternalNode_cpu /* (N + 255) / 256, 256 */ (internalNodes_h, N);

	buildBVH_cpu /* (N + 255) / 256, 256 */ (morton_h, leafNodes_h, internalNodes_h, N);
	//checkNodes_cpu /* (N + 255) / 256, 256 */ (internalNodes_h, N);

	traverseBVH_cpu /* (N + 255) / 256, 256 */ (leafNodes_h, internalNodes_h, index_h, Nencpairs2_h, Encpairs2_h, N_h[0], N);
}

