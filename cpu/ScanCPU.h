
//**************************************
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of multiple blocks
//must be followed by Scan32d2 and Scan32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************



//**************************************
//This kernel reads the result from the multiple thread block kernel Scan32d1
//and performs the last summation step in
// --a single thread block --
//
//must be followed by Scan32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************


//**************************************
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of multiple warps, but only 1 thread block
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************

//**************************************
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of only 1 single warp
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************

#if def_CPU == 1
void Scan_cpu(int2 *scan_h, int *Encpairs3_h, int *Nencpairs3_h, const int N, const int NencMax){

	for(int id = 0; id < N; ++id){
		if(id > 0){
			scan_h[id].x += scan_h[id - 1].x;
		}

		if(Encpairs3_h[id * NencMax + 0] > 0){
			int t1 = scan_h[id].x;
			Encpairs3_h[(t1 - 1) * NencMax + 1] = id;
		}
	}

	Nencpairs3_h[0] = scan_h[N - 1].x;
}
#endif

