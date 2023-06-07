#include <cstdio>
#include <cstdlib>

__global__ void calc_b(float *b, float *u, float *v,float rho, float dt, float dx, float dy, int nx, int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0 || i>=nx-1) return;
    if(j==0 || j>=ny-1) return;   
    b[j*ny + i*nx] = rho * (1/dt *
                   ((u[j*ny + (i+1)*nx] - u[j*ny + (i-1)*nx]) / (2 * dx)  +  (v[(j+1)*ny + i*nx] - v[(j-1)*ny + i*nx]) / (2* dy)) -
                   ((u[j*ny + (i+1)*nx] - u[j*ny + (i-1)*nx]) / (2 * dx)) * ((u[j*ny + (i+1)*nx] - u[j*ny + (i-1)*nx]) / (2 * dx)) - 
                     2 * ((u[(j+1)*ny + i*nx] - u[(j-1)*ny + i*nx]) / (2 * dy) *
                    (v[j*ny + (i+1)*nx] - v[j*ny + (i-1)*nx]) / (2 * dx)) - 
                   ((v[(j+1)*ny + i*nx] - v[(j-1)*ny + i*nx]) / (2 * dy)) * 
                   ((v[(j+1)*ny + i*nx] - v[(j-1)*ny + i*nx]) / (2 * dy)));
}

__global__ void calc_p(float *b, float *p, float *pn, float dx, float dy, int nx, int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0 || i>=nx-1) return;
    if(j==0 || j>=ny-1) return;   
    p[j*ny + i*nx] = (dy*dy * (pn[j*ny + (i+1)*nx] + pn[j*ny + (i-1)*nx]) +
                      dx*dx * (pn[(j+1)*ny + i*nx] + pn[(j-1)*ny + i*nx]) -
                               b[j*ny + i*nx] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
}

__global__ void calc_uv(float *u, float *v, float *un, float *vn, float *p, float rho, float nu, float dt, float dx, float dy, int nx, int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0 || i>=nx-1) return;
    if(j==0 || j>=ny-1) return;
    u[j*ny + i*nx] = un[j*ny + i*nx] - un[j*ny + i*nx] * dt / dx * (un[j*ny + i*nx] - un[j*ny + (i-1)*nx])
                   - un[j*ny + i*nx] * dt / dy * (un[j*ny + i*nx] - un[(j-1)*ny + i*nx])
                   - dt / (2*rho*dx) * (p[j*ny + (i+1)*nx] - p[j*ny + (i-1)*nx])
                   + nu * dt / (dx*dx) * (un[j*ny + (i+1)*nx] - 2 * un[j*ny + i*nx] + un[j*ny + (i-1)*nx])
                   + nu * dt / (dy*dy) * (un[(j+1)*ny + i*nx] - 2 * un[j*ny + i*nx] + un[(j-1)*ny + i*nx]);
    v[j*ny + i*nx] = vn[j*ny + i*nx] - vn[j*ny + i*nx] * dt / dx * (vn[j*ny + i*nx] - vn[j*ny + (i-1)*nx])
                   - vn[j*ny + i*nx] * dt / dy * (vn[j*ny + i*nx] - vn[(j-1)*ny + i*nx])
                   - dt / (2*rho*dx) * (p[(j+1)*ny + i*nx] - p[(j-1)*ny + i*nx])
                   + nu * dt / (dx*dx) * (vn[j*ny + (i+1)*nx] - 2 * vn[j*ny + i*nx] + vn[j*ny + (i-1)*nx])
                   + nu * dt / (dy*dy) * (vn[(j+1)*ny + i*nx] - 2 * vn[j*ny + i*nx] + vn[(j-1)*ny + i*nx]);
}   

int main(){
    int   nx = 41;
    int   ny = 41;
    int   nt = 500;
    int   nit = 50;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = .01;
    float rho = 1;
    float nu = .02;
    float *u;
    float *v;
    float *p;
    float *b;
    cudaMallocManaged(&u, nx*ny*sizeof(float));
    cudaMallocManaged(&v, nx*ny*sizeof(float));
    cudaMallocManaged(&p, nx*ny*sizeof(float));
    cudaMallocManaged(&b, nx*ny*sizeof(float));
    // vector for copy
    float *un;
    float *vn;
    float *pn;
    cudaMallocManaged(&un, nx*ny*sizeof(float));
    cudaMallocManaged(&vn, nx*ny*sizeof(float));
    cudaMallocManaged(&pn, nx*ny*sizeof(float));
    // cavity
    for(int n=0; n<nt; n++){
        printf("%d/%d\n",n, nt);
        // calculate b
        calc_b<<<1, ny*nx>>>(b, u, v, rho, dt, dx, dy, nx, ny); 
        cudaDeviceSynchronize();
        // calculate p
        for(int it=0; it<nit; it++){
            // pn = p.copy()
            for(int j=0; j<ny; j++){
                for(int i=0; i<nx; i++){
                    pn[j*ny + i*nx] = p[j*ny + i*nx];
                }
            }
            calc_p<<<1, ny*nx>>>(b, p, pn, dx, dy, nx, ny);
            cudaDeviceSynchronize();
            for(int j=0; j<ny; j++){
                p[j*ny + (nx-1)*nx] = p[j*ny + (nx-2)*nx];  // p[:, -1] = p[:, -2]
                p[j*ny + 0*nx]      = p[j*ny + 1*nx];       // p[:,  0] = p[:,  1]
            }
            for(int i=0; i<nx; i++){
                p[0*ny + i*nx]  = p[1*ny + i*nx];
                p[(ny-1)*ny + i*nx] = 0;
            }
        }
        // calculate u, v
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                un[j*ny + i*nx] = u[j*ny + i*nx];
                vn[j*ny + i*nx] = v[j*ny + i*nx];
            }
        }     
        calc_uv<<<1, ny*nx>>>(u, v, un, vn, p, rho, nu, dt, dx, dy, nx, ny);
        cudaDeviceSynchronize();
        for(int j=0; j<ny; j++){
            u[j*ny +      0*nx] = 0;  // u[:, 0] = 0;
            u[j*ny + (nx-1)*nx] = 0;  // u[:,-1] = 0;
            v[j*ny +      0*nx] = 0;  
            v[j*ny + (nx-1)*nx] = 0;  
        }
        for(int i=0; i<nx; i++){
            u[0*ny      + i*nx] = 0;  // u[0, :] = 0;
            u[(ny-1)*ny + i*nx] = 1;  // u[-1,:] = 1;
            v[0*ny      + i*nx] = 0;
            v[(ny-1)*ny + i*nx] = 0;
        }
    }
    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            printf("u[%d][%d]=%f\n", j, i, u[j*ny + i*nx]);
        }
    }
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
}

