// #include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include <chrono>
#include <ctime>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>


__global__ void initializeParticleKernel(
        float* d_particlesPositions,
        float* d_particlesVelocities,
        float* d_personalBestValues,
        float* d_personalBestPositions,
        float* d_globalBestPositions,
        float* d_globalBestValues,
        int Q_size, 
        int numParticles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(idx, 0, 0, &state);
    if (idx < numParticles) {
        // Initialize position and velocity of particles
        for (int i = 0; i < Q_size; i++) {
            int posIdx = idx * Q_size + i;
            float randVal = curand_uniform(&state);
            d_particlesPositions[posIdx] = randVal; 
            d_particlesVelocities[posIdx] = 0.0f;
            d_personalBestPositions[posIdx] = d_particlesPositions[posIdx];
            d_globalBestPositions[posIdx] = 0.0f;
        }
        d_personalBestValues[idx * 2] = INFINITY;
        d_personalBestValues[idx * 2 + 1] = INFINITY;
        d_globalBestValues[idx * 2] = INFINITY;
        d_globalBestValues[idx * 2 + 1] = INFINITY;
        
    }
}


// // Function to perform matrix-vector multiplication (Qx) using BLAS
// std::vector<std::vector<float>> multiplyQXBLAS(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& X, int Q_size, int numParticles) {
//     // Flatten the input matrices Q and X for CBLAS
//     std::vector<float> Q_flat(Q_size * Q_size);
//     std::vector<float> X_flat(Q_size * numParticles);
//     for (int i = 0; i < Q_size; ++i) {
//         for (int j = 0; j < Q_size; ++j) {
//             Q_flat[i * Q_size + j] = Q[i][j];
//         }
//     }
//     for (int i = 0; i < Q_size; ++i) {
//         for (int j = 0; j < numParticles; ++j) {
//             X_flat[i * numParticles + j] = X[i][j];
//         }
//     }

//     // Allocate space for the result matrix Y
//     std::vector<float> Y_flat(Q_size * numParticles);

//     // Perform the matrix multiplication using CBLAS
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                 Q_size, numParticles, Q_size, // dimensions of matrices
//                 1.0, // alpha
//                 &Q_flat[0], Q_size, // A and its leading dimension
//                 &X_flat[0], numParticles, // B and its leading dimension
//                 0.0, // beta
//                 &Y_flat[0], numParticles); // C and its leading dimension

//     // Convert the flat result matrix back into a vector of vectors
//     std::vector<std::vector<float>> Y(Q_size, std::vector<float>(numParticles));
//     for (int i = 0; i < Q_size; ++i) {
//         for (int j = 0; j < numParticles; ++j) {
//             Y[i][j] = Y_flat[i * numParticles + j];
//         }
//     }

//     return Y;
// }

// // Objective function f1 calculation (x'Qx = inner product of ith column of Y with ith column of X)
// float f1(const std::vector<std::vector<float>>& Y, const std::vector<std::vector<float>>& X, size_t particleIndex) 
// {
//     float result = 0.0f;
//     // Assuming that Y and X have the same number of rows as the dimension of the problem
//     // and the same number of columns as the number of particles.
//     for (size_t i = 0; i < X.size(); i++) { // X.size() should be equal to the dimension of the problem
//         result += Y[i][particleIndex] * X[i][particleIndex];
//     }
//     return result;
// }
// // Overload for std::vector<int>
// float f1(const std::vector<std::vector<float>>& Y, const std::vector<std::vector<int>>& X, size_t particleIndex) 
// {
//     float result = 0.0f;
//     // Assuming that Y and X have the same number of rows as the dimension of the problem
//     // and the same number of columns as the number of particles.
//     for (size_t i = 0; i < X.size(); i++) { // X.size() should be equal to the dimension of the problem
//         result += Y[i][particleIndex] * X[i][particleIndex];
//     }
//     return result;
// }

// // Objective function f2 calculation (Sum(xi * (1 - xi)))
// float f2(const std::vector<float>& x) 
// {
//     float result = 0.0f;
//     for (float xi : x) result += xi * (1.0f - xi);
//     return result;
// }

// Update the velocity and position of the particle according to PSO rules
// void updateParticle(Particle& p, const std::vector<float>& gBestPosition, float w, float c1, float c2, int dim) {
//     // Velocity and position update based on PSO equations
//     for (int i = 0; i < dim; i++) {
//         float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for cognitive component
//         float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for social component
        
//         // Update velocity for each dimension
//         p.velocity[i] = w * p.velocity[i] +                                      
//                         c1 * r1 * (p.personalBestPosition[i] - p.position[i]) +  
//                         c2 * r2 * (gBestPosition[i] - p.position[i]);            

//         // Update position for each dimension
//         p.position[i] += p.velocity[i];
        
//          // Clamping to ensure position is within [0,1]
//         if (p.position[i] < 0.0) p.position[i] = 0.0;
//         if (p.position[i] > 1.0) p.position[i] = 1.0;
//     }
// }

// // Function to check if solution 'a' dominates solution 'b'
// bool dominates(const std::vector<float>& a, const std::vector<float>& b) 
// {
//     bool anyBetter = false;
//     for (int i = 0; i < 2; ++i) 
//     {
//         if (a[i] > b[i]) return false; // 'a' can't be worse in any objective
//         if (a[i] < b[i]) anyBetter = true; // 'a' has to be better in at least one objective
//     }
//     return anyBetter;
// }

int main() 
{
    // Initialize random seed
    unsigned int seed = static_cast<unsigned>(time(nullptr));
    srand(seed);

    // Define parameters as in the paper
    const float w = 0.729;   // Inertia weight
    const float c1 = 1.4955; // Cognitive weight
    const float c2 = 1.4955; // Social weight
    const int t_max = 10000; // Maximum number of iterations
    const int numParticles = 128; // Number of particles swarm, may vary

    // Define the Q matrix, its size and result value of real solution
    // Examples are from page 9 of 'A Tutorial on Formulating and Using QUBO Models'
    
    // std::vector<std::vector<float>> Q = 
    // {
    //     {-3525, 175, 325, 775, 1050, 425, 525, 250},
    //     {175, -1113, 91, 217, 294, 119, 147, 70},
    //     {325, 91, -1989, 403, 546, 221, 273, 130},
    //     {775, 217, 403, -4185, 1302, 527, 651, 310},
    //     {1050, 294, 546, 1302, -5208, 714, 882, 420},
    //     {425, 119, 221, 527, 714, -2533, 357, 170},
    //     {525, 147, 273, 651, 882, 357, -3045, 210},
    //     {250, 70, 130, 310, 420, 170, 210, -1560}
    // };

    // exmple from page 34
    int Q_size = 10;
    float Q[Q_size * Q_size] = {
        -526, 150, 160, 190, 180, 20, 40, -30, -60, -120,
        150, -574, 180, 200, 200, 20, 40, -30, -60, -120,
        160, 180, -688, 220, 200, 40, 80, -20, -40, -80,
        190, 200, 220, -645, 240, 30, 60, -40, -80, -160,
        180, 200, 200, 240, -605, 20, 40, -40, -80, -160,
        20, 20, 40, 30, 20, -130, 20, 0, 0, 0,
        40, 40, 80, 60, 40, 20, -240, 0, 0, 0,
        -30, -30, -20, -40, -40, 0, 0, 110, 20, 40,
        -60, -60, -40, -80, -80, 0, 0, 20, 240, 80,
        -120, -120, -80, -160, -160, 0, 0, 40, 80, 560
    };
    
    // const float fOptimal = -6889.0f; // Known optimal objective value for comparison
    const float fOptimal = -916.0f;

    //  For timing
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();



    /************************************ Position Initialization *********************************/
    /************************************ Algorith 1 line 1 - 3 ***********************************/
    // Initialize particles and global best
    // Note: All are flattened in row-major way
    float* h_particlesPositions = new float[numParticles * Q_size];
    float* h_particlesVelocities = new float[numParticles * Q_size];
    float* h_personalBestValues = new float[2 * numParticles];
    float* h_personalBestPositions = new float[numParticles * Q_size];
    float* h_globalBestPositions = new float[numParticles * Q_size];
    float* h_globalBestValues = new float[2 * numParticles];

    float* d_particlesPositions;
    float* d_particlesVelocities;
    float* d_personalBestValues;
    float* d_personalBestPositions;
    float* d_globalBestPositions;
    float* d_globalBestValues;

    // Allocate particles on device
    // Allocate memory for particle's position, velocity, and personal and global BestPosition arrays
    cudaMalloc(&d_particlesPositions, numParticles * Q_size * sizeof(float));
    cudaMalloc(&d_particlesVelocities, numParticles * Q_size * sizeof(float));
    cudaMalloc(&d_personalBestValues, 2 * numParticles * sizeof(float));
    cudaMalloc(&d_personalBestPositions, numParticles * Q_size * sizeof(float));
    cudaMalloc(&d_globalBestPositions, numParticles * Q_size * sizeof(float));
    cudaMalloc(&d_globalBestValues, 2 * numParticles * sizeof(float));

    // Calculate number of blocks and threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    initializeParticleKernel<<<blocksPerGrid, threadsPerBlock>>>(    
        d_particlesPositions,
        d_particlesVelocities,
        d_personalBestValues,
        d_personalBestPositions,
        d_globalBestPositions,
        d_globalBestValues,
        Q_size, 
        numParticles
    );

    // Copy back the particles to host
    cudaMemcpy(h_particlesPositions, d_particlesPositions, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particlesVelocities, d_particlesVelocities, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_personalBestValues, d_personalBestValues, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_personalBestPositions, d_personalBestPositions, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_globalBestValues, d_globalBestValues, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_globalBestPositions, d_globalBestPositions, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);

    /******************************** End of Position Initialization ********************************/



    /******************************************** Main Loop ***************************************/
    /************************************ Algorith 1 line 4 - 15 **********************************/
    // Allocate space for Y and Q
    float* h_Y = new float[Q_size * numParticles];
    float* d_Y;
    float* d_Q;
    cudaMalloc(&d_Y, Q_size * numParticles * sizeof(float));
    cudaMalloc(&d_Q, Q_size * Q_size * sizeof(float));
    cudaMemcpy(d_Y, h_Y, Q_size * Q_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Initialize CUBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization error!!!!!!!!\n";
        return EXIT_FAILURE;
    }
    int m = numParticles, n = Q_size, k = Q_size;
    int lda = m, ldb = n, ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int t = 1; t <= t_max; ++t) 
    {
        // Perform the matrix multiplication Y = QX
        // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
        // h_particlesPositions = X^T
        // In CPU: Y = Q x X = Q x h_particlesPositions^T
        // But in GPU: we actually get on host (Q^T x X^T)^T = ((X x Q)^T)^T = X x Q
        // Therefor we have to do in cuBLAS Y^T = X^T x Q^T where C = Y^T, A = X^T = h_particlesPositions, B = Q^T
        // For cuBlas, that is we do X x Q = h_particlesPositions^T x Q on it, so h_particlesPositions need transpose
        status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, h_particlesPositions, lda, d_Q, ldb, &beta, d_Y, ldc);
        cudaMemcpy(h_Y, d_Y, Q_size * numParticles * sizeof(float), cudaMemcpyDeviceToHost);
        // for (size_t particleIndex = 0; particleIndex < particles.size(); ++particleIndex)
        // {
        //     auto& p = particles[particleIndex];

        //     // Calculate objective function values
        //     std::vector<float> objectives = {f1(Y, X, particleIndex), f2(p.position)};

        //     if (dominates(objectives, p.personalBestValues)) 
        //     {
        //         p.personalBestPosition = p.position;
        //         p.personalBestValues = objectives;
        //     }
        // }

        // // Build the archive from non-dominated personal bests relative to the first particle's personal best
        // std::vector<Particle> archive;
        // for (const auto& p : particles) 
        // {
        //     if (!dominates(particles[0].personalBestValues, p.personalBestValues)) archive.push_back(p);
        // }

        // // Update velocities and positions based on the archive
        // for (auto& p : particles) 
        // {
        //     if (!archive.empty()) 
        //     {
        //         int randomIndex = rand() % archive.size();
        //         const auto& globalBest = archive[randomIndex];
        //         updateParticle(p, globalBest.personalBestPosition, w, c1, c2, Q_size);
        //     }
        // }
    }
    
    // Stop recording
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GPU execution time: " << duration / 1000000.0f << " seconds\n";


    // // Round solutions and find the one with minimal f1
    // std::vector<std::vector<int>> roundedSolutions(Q_size, std::vector<int>(numParticles, 0));
    // // Round solutions and get Y
    // for (size_t particleIndex = 0; particleIndex < numParticles; ++particleIndex) 
    // {
    //     auto& p = particles[particleIndex];
    //     for (int i = 0; i < Q_size; ++i) 
    //     {
    //         roundedSolutions[i][particleIndex] = std::round(p.position[i]);
    //     }
    // }
    // std::vector<std::vector<float>> Y_rounded = multiplyQX(Q, roundedSolutions, Q_size, numParticles);

    // // Find the one with minimal f1
    // float minimalF1 = std::numeric_limits<float>::max();
    // std::vector<int> bestRoundedSolution(Q_size); // This will store the best rounded solution

    // for (size_t particleIndex = 0; particleIndex < numParticles; ++particleIndex)
    // {
    //     float currentF1 = f1(Y_rounded, roundedSolutions, particleIndex);
    //     if (currentF1 < minimalF1) 
    //     {
    //         minimalF1 = currentF1;
    //         // Update bestRoundedSolution with integer values
    //         bestRoundedSolution = roundedSolutions[particleIndex];
    //     }
    // }

    // // Calculate relative error for accuracy of found solutions
    // float relativeError = (minimalF1 - fOptimal) / std::abs(fOptimal);

    // // Output the best rounded solution and its f1 value
    // std::cout << "Best solution found:" << std::endl;
    // for (int val : bestRoundedSolution) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\nRelative error compared to real optimal f1: " << relativeError * 100 << "%" << std::endl;


    // Clean up
    // Destroy CUBLAS handle
    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_particlesPositions);
    cudaFree(d_particlesVelocities);
    cudaFree(d_personalBestPositions);
    cudaFree(d_personalBestValues);
    cudaFree(d_globalBestPositions);
    cudaFree(d_globalBestValues);
    cudaFree(d_Y);
    cudaFree(d_Q);

    // Free host memory
    delete[] h_particlesPositions;
    delete[] h_particlesVelocities;
    delete[] h_personalBestPositions;
    delete[] h_personalBestValues;
    delete[] h_globalBestPositions;
    delete[] h_globalBestValues;

}