#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_sort(int *bucket,int *key, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[i] = 0;
  atomicAdd(&bucket[key[i]], 1);
  int sum = 0;
  for (int j=0; j<range; j++){
    if ((sum <= i) && (i < sum+bucket[j])){
      key[i] = j;
      return;
    }
    sum += bucket[j];
  }
}


int main(){
  int n = 50;
  int range = 4;
  // cuda
  int *bucket;
  int *key;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  bucket_sort<<<1,n>>>(bucket, key, range);
  cudaDeviceSynchronize();
  
  printf("\n");
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(bucket);
  cudaFree(key);
}
