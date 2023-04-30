#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);

  for(int i=0; i<N; i++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);

    // culculate 1/r
    __m256 rxvec = _mm256_sub_ps(xi, xvec);
    __m256 ryvec = _mm256_sub_ps(yi, yvec);
    __m256 rx2 = _mm256_mul_ps(rxvec, rxvec);
    __m256 ry2 = _mm256_mul_ps(ryvec, ryvec);
    __m256 r2vec = _mm256_add_ps(rx2, ry2);
    // mask
    __m256 mask = _mm256_cmp_ps(xvec, xi, _CMP_EQ_OQ);
    __m256 rinv = _mm256_rsqrt_ps(r2vec);
    __m256 zero = _mm256_set1_ps(0);
    rinv = _mm256_blendv_ps(rinv, zero, mask);
    __m256 rinv3 = _mm256_mul_ps(rinv, rinv);
    rinv3 = _mm256_mul_ps(rinv3, rinv);
 
    __m256 fxvec = _mm256_mul_ps(rxvec, mvec);
    __m256 fyvec = _mm256_mul_ps(ryvec, mvec);
    fxvec = _mm256_mul_ps(fxvec, rinv3);
    fyvec = _mm256_mul_ps(fyvec, rinv3);
    
    // reduction
    __m256 fxsum = _mm256_permute2f128_ps(fxvec, fxvec, 1);
    __m256 fysum = _mm256_permute2f128_ps(fyvec, fyvec, 1);
    fxsum = _mm256_add_ps(fxsum, fxvec);
    fxsum = _mm256_hadd_ps(fxsum, fxsum);
    fxsum = _mm256_hadd_ps(fxsum, fxsum);

    fysum = _mm256_add_ps(fysum, fyvec);
    fysum = _mm256_hadd_ps(fysum, fysum);
    fysum = _mm256_hadd_ps(fysum, fysum); 
    _mm256_store_ps(fx, fxsum);
    _mm256_store_ps(fy, fysum);
    printf("%d %g %g \n",i, -fx[0], -fy[0]);
  
    /*
    for(int j=0; j<N; j++) {  // vectorize  
      if(i != j) {  // use mask
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];  // use _mm2565_sub_ps
        float r = std::sqrt(rx * rx + ry * ry);  // use mm256_rsqrt_ps
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
    */
  }
}
