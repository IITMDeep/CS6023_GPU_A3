#include <iostream>
#include <vector>
#include <cuda.h>
#include <climits>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;

#define INF INT_MAX
#define NORM 1000000007
#define BLOCK_SIZE 1024  // Optimal block size for GPU kernels

// Edge structure to represent graph edges
struct Edge {
    int u, v, wt, type;  // u: source, v: destination, wt: weight, type: terrain type
};


// Device function to find root of a vertex with path compression
__device__ int findRoot(int *component, int vertex) {
    int root = vertex;
    while (root != component[root]) {
        int child = component[root];
        int parent = component[component[root]];
        component[root] = parent; // path compression
        root = child;
    }
    return root;
}

// Kernel to initialize minEdge and minWeight arrays
__global__ void initArrays(int *minEdge, int *minWeight, int V) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    else {
        minEdge[tid] = -1;  // -1 indicates no edge found yet
        minWeight[tid] = INF;  // Initialize with infinity
    }
}

// Kernel to find minimum weight edges for each component
__global__ void findMinEdges(Edge *edges, int *minWeight, int *component, int E) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= E) return;
    else {
      Edge e = edges[tid];
      int u = e.u;
      int v = e.v;

      int comp_u = findRoot(component, u);
      int comp_v = findRoot(component, v);

      if (comp_u == comp_v) return;  // Skip if same component
      else{
        int weight = e.wt * e.type; // Apply terrain type multiplier to weight

        // Atomically update minimum weights for both components
        atomicMin(&minWeight[comp_u], weight);
        atomicMin(&minWeight[comp_v], weight);
      }
    }
}


// Kernel to merge components using the found minimum edges
__global__ void mergeComponents(Edge *edges, int *minEdge, int *component, int *addedToMST, long long *mstWeight, int *anyMerge, int V) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= V) return;
    else{
      int edgeIdx = minEdge[tid];
      if (edgeIdx == -1) return;  // No edge to merge
      else{
        Edge e = edges[edgeIdx];
        int u = e.u;
        int v = e.v;  

        int rootU = findRoot(component, u);
        int rootV = findRoot(component, v);

        if (rootU == rootV) return;  // Already in same component
        else{
          int small = min(rootU, rootV);
          int large = max(rootU, rootV);

          // Atomic compare-and-swap to merge components
          if (atomicCAS(&component[large], large, small) == large) {
              int weight = e.wt * e.type; // Apply terrain type multiplier

              atomicExch(&addedToMST[edgeIdx], 1);  // Mark edge as added to MST
              atomicAdd((unsigned long long int *)mstWeight, weight);  // Update MST weight

              *anyMerge = 1;  // Set flag indicating a merge occurred
          }
        }
      }
    }
}

// Kernel to set the minimum edge indices for each component
__global__ void setMinEdges(Edge *edges, int *minEdge, int *minWeight, int *component, int E) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= E) return;
    else{
      Edge e = edges[tid];
      int u = e.u;
      int v = e.v;
      
      int comp_u = findRoot(component, u);
      int comp_v = findRoot(component, v);

      if (comp_u == comp_v) return;  // Skip if same component
      else{
        int weight = e.wt * e.type; // Apply terrain type multiplier

        // Atomically set the minimum edge index if weight matches
        if (weight == minWeight[comp_u]){
          atomicExch(&minEdge[comp_u], tid);
        }
        if (weight == minWeight[comp_v]){
          atomicExch(&minEdge[comp_v], tid);
        }
      }
    }
}

// Kernel to perform path compression and handle large MST weights
__global__ void compressComponents(int *component, int V, long long *mstWeight) {
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // if (tid < V) findRoot(component, tid);  // Path compression for all vertices

    // Modulo operation for large MST weights
    *mstWeight %= NORM;
}

// Kernel to initialize component array
__global__ void initComponent(int *d_component, int V){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid > V) return;
  else {
    d_component[tid] = tid;  // Each vertex is its own component initially
  }
}



int main() {
    int V, E;
    cin >> V >> E;
    vector<Edge> h_edges(E);
    for (int i = 0; i < E; i++) {
        int u, v, wt;
        int x = 1;
        string type;
        cin >> u >> v >> wt >> type;
        // Set terrain type multiplier
        if (type == "green") x = 2;
        else if (type == "traffic") x = 5;
        else if (type == "dept") x = 3;
        h_edges[i] = {u, v, wt, x};
    }


    // Main Boruvka's algorithm implementation on GPU
    Edge *d_edges;
    int *d_component, *d_minEdge, *d_minWeight, *d_addedToMST, *d_anyMerge;
    long long *d_mstWeight;

    // Allocate GPU memory
    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_component, V * sizeof(int));
    cudaMalloc(&d_minEdge, V * sizeof(int));
    cudaMalloc(&d_minWeight, V * sizeof(int));
    cudaMalloc(&d_addedToMST, E * sizeof(int));
    cudaMalloc(&d_mstWeight, sizeof(long long));
    cudaMalloc(&d_anyMerge, sizeof(int));

    // Copy edges from host to device
    cudaMemcpy(d_edges, h_edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    
    // Initialize device memory
    cudaMemset(d_addedToMST, 0, E * sizeof(int));
    cudaMemset(d_mstWeight, 0, sizeof(int));
    
    // Calculate grid sizes for kernel launches
    int Eblocks = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int Vblocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;


    // Start timer
    auto start = chrono::high_resolution_clock::now();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Initialize components
    initComponent<<<Vblocks , BLOCK_SIZE>>>(d_component, V);    
    for (int iter = 0; iter < 100; ++iter) {
        cudaMemset(d_anyMerge, 0, sizeof(int));

        // Execute Boruvka steps
        initArrays<<<Vblocks, BLOCK_SIZE>>>(d_minEdge, d_minWeight, V);
        findMinEdges<<<Eblocks, BLOCK_SIZE>>>(d_edges, d_minWeight, d_component, E);
        setMinEdges<<<Eblocks, BLOCK_SIZE>>>(d_edges, d_minEdge, d_minWeight, d_component, E);
        mergeComponents<<<Vblocks, BLOCK_SIZE>>>(d_edges, d_minEdge, d_component, d_addedToMST, d_mstWeight, d_anyMerge, V);
        
        // Check if any merge occurred
        int anyMerge = 0;
        cudaMemcpy(&anyMerge, d_anyMerge, sizeof(int), cudaMemcpyDeviceToHost);
        if (!anyMerge) break;  // Terminate if no merges (MST complete)
    }
    compressComponents<<<1, 1>>>(d_component, V, d_mstWeight);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // End timer
    auto end = chrono::high_resolution_clock::now();

    // Copy final MST weight back to host
    int mstWeight = 0;
    cudaMemcpy(&mstWeight, d_mstWeight, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_edges); cudaFree(d_component); cudaFree(d_minEdge);
    cudaFree(d_minWeight); cudaFree(d_addedToMST); cudaFree(d_mstWeight); cudaFree(d_anyMerge);

    cout<<mstWeight<<endl;
    chrono::duration<double> elapsed = end - start;

    // Write results to files
    // ofstream fout("cuda.out");
    // fout << mstWeight << "\n"; fout.close();

    // ofstream ftime("cuda_timing.out");
    // ftime << elapsed.count() << "\n"; ftime.close();
    // return 0;
}