#include <curand_kernel.h>
#include <cublas_v2.h>

#include <chrono>
#include <ctime>
#include <limits>
#include <cstdlib>
#include <cmath>

#include <chrono>
#include <vector>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


__global__ void initializeParticleKernel(
        float* d_particlesPositions,
        float* d_particlesVelocities,
        float* d_personalBestValues,
        float* d_personalBestPositions,
        float* d_globalBestPositions,
        float* d_globalBestValues,
        int Q_size, 
        int numParticles) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(idx, 0, 0, &state);
    if (idx < numParticles) {
        // Initialize position and velocity of particles
        for (int i = 0; i < Q_size; i++) {
            int posIdx = idx * Q_size + i;
            float randVal = curand_uniform(&state);
            d_particlesPositions[posIdx] = randVal; 
            // printf("Random value at idx %d, i %d: %f\n", idx, i, randVal);
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

__global__ void computeFsUpdatePersonalBest(
        float* d_particlesPositions,
        float* d_personalBestValues,
        float* d_personalBestPositions,
        float* d_Y,
        int Q_size, 
        int numParticles) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numParticles) {
        // Calculate new target values
        float f1 = 0.0f, f2 = 0.0f;
        for (int j = 0; j < Q_size; j++) {
            float xj = d_particlesPositions[idx * Q_size + j];
            f1 += xj * d_Y[j * numParticles + idx];
            f2 += xj * (1 - xj);
        }
        // Check if new values dominate current personal best
        float personalBestf1 = d_personalBestValues[idx * 2];
        float personalBestf2 = d_personalBestValues[idx * 2 + 1];
        if (f1 < personalBestf1 || f2 < personalBestf2) {
            if (f1 <= personalBestf1 && f2 <= personalBestf2) {
                d_personalBestValues[idx * 2] = f1;
                d_personalBestValues[idx * 2 + 1] = f2;
                for (int j = 0; j < Q_size; j++) {
                    int posIdx = idx * Q_size + j;
                    d_personalBestPositions[posIdx] = d_particlesPositions[posIdx];
                }
            }
        }     
    }
}

// Update the velocity and position of the particle according to PSO rules
__global__ void updateVelocityAndPosition(
        float* d_archivePersonalBestPositions,
        float* d_archivePersonalBestValues,
        float* d_particlesPositions,
        float* d_particlesVelocities,
        float* d_personalBestValues,
        float* d_personalBestPositions,
        float* d_globalBestPositions,
        float* d_globalBestValues,
        int Q_size, 
        int numParticles,
        int archiveSize,
        float w,
        float c1, 
        float c2) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numParticles) 
    {
        curandState state;
        curand_init(idx, 0, 0, &state);
        float r1 = curand_uniform(&state); // Random factor for cognitive component
        float r2 = curand_uniform(&state); // Random factor for social component
        // Randomly pick one from archive
        int bestIdx = curand(&state) % archiveSize;
        // printf("Random value at idx %d, i %d\n", idx, bestIdx);
        for (int j = 0; j < Q_size; ++j)
        {
            // Get indice for thr current particle and the one used as global best
            int k_x = idx * Q_size + j;
            int k_best = bestIdx * Q_size + j;
            // Update velocity for each dimension
            d_particlesVelocities[k_x] = w * d_particlesVelocities[k_x] + 
                                         c1 * r1 * (d_personalBestPositions[k_x] - d_particlesPositions[k_x]) + 
                                         c2 * r2 * (d_archivePersonalBestPositions[k_best] - d_particlesPositions[k_x]);
            // Update position for each dimension
            d_particlesPositions[k_x] += d_particlesVelocities[k_x];
            if (d_particlesPositions[k_x] < 0.0) d_particlesPositions[k_x] = 0.0;
            else if (d_particlesPositions[k_x] > 1.0) d_particlesPositions[k_x] = 1.0;
        }
    }
}

// Function to read the matrix from the file
bool readMatrix(const std::string& filePath, std::vector<float>& Q, int& Q_size) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    std::string line;

    // Read the first line for the matrix size
    if (!std::getline(file, line)) {
        std::cerr << "Failed to read the matrix size from the file." << std::endl;
        return false;
    }
    Q_size = std::stoi(line);

    // Read the matrix data
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float number;
        while (iss >> number) {
            Q.push_back(number);
        }
    }

    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    std::vector<float> Q;
    int Q_size = 0;

    if (!readMatrix(filePath, Q, Q_size)) {
        std::cerr << "Error reading matrix from file." << std::endl;
        return 1;
    }

    std::cout << "Q_size: " << Q_size << std::endl;
    // std::cout << "Matrix Q:" << std::endl;
    // for (int i = 0; i < Q_size; ++i) {
    //     for (int j = 0; j < Q_size; ++j) {
    //         std::cout << Q[i * Q_size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Initialize random seed
    unsigned int seed = static_cast<unsigned>(time(nullptr));
    srand(seed);

    // Define parameters as in the paper
    const float w = 0.729;   // Inertia weight
    const float c1 = 1.4955; // Cognitive weight
    const float c2 = 1.4955; // Social weight
    const int t_max = 1; // Maximum number of iterations
    const int numParticles = 64; // Number of particles swarm, may vary

    // Define the Q matrix, its size and result value of real solution
    // Examples are from page 9 of 'A Tutorial on Formulating and Using QUBO Models'
    
    // int Q_size = 8;
    // std::vector<float> Q = 
    // {
    //     -3525, 175, 325, 775, 1050, 425, 525, 250,
    //     175, -1113, 91, 217, 294, 119, 147, 70,
    //     325, 91, -1989, 403, 546, 221, 273, 130,
    //     775, 217, 403, -4185, 1302, 527, 651, 310,
    //     1050, 294, 546, 1302, -5208, 714, 882, 420,
    //     425, 119, 221, 527, 714, -2533, 357, 170,
    //     525, 147, 273, 651, 882, 357, -3045, 210,
    //     250, 70, 130, 310, 420, 170, 210, -1560
    // };
    // const float fOptimal = -6889.0f; // Known optimal objective value for comparison

    // exmple from page 34
    // int Q_size = 10;

    // std::vector<float> Q = {
    //     -526, 150, 160, 190, 180, 20, 40, -30, -60, -120,
    //     150, -574, 180, 200, 200, 20, 40, -30, -60, -120,
    //     160, 180, -688, 220, 200, 40, 80, -20, -40, -80,
    //     190, 200, 220, -645, 240, 30, 60, -40, -80, -160,
    //     180, 200, 200, 240, -605, 20, 40, -40, -80, -160,
    //     20, 20, 40, 30, 20, -130, 20, 0, 0, 0,
    //     40, 40, 80, 60, 40, 20, -240, 0, 0, 0,
    //     -30, -30, -20, -40, -40, 0, 0, 110, 20, 40,
    //     -60, -60, -40, -80, -80, 0, 0, 20, 240, 80,
    //     -120, -120, -80, -160, -160, 0, 0, 40, 80, 560
    // };
    // const float fOptimal = -916.0f;

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
    cudaDeviceSynchronize();
    // Check for any errors launching the kernel
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed in initializeParticleKernel: %s\n", cudaGetErrorString(error));
    }
	
    // // Check initialization
    // std::cout << "positions:" << std::endl;
    // for (int i = 0; i < numParticles; ++i) {
    //     for (int j = 0; j < Q_size; ++j) {
    //     std::cout << h_particlesPositions[i * Q_size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    /******************************** End of Position Initialization ********************************/



    /******************************************** Main Loop ***************************************/
    /************************************ Algorith 1 line 4 - 15 **********************************/
    // Allocate space for Y and Q
    float* d_Y;
    float* d_Q;
    cudaMalloc(&d_Y, Q_size * numParticles * sizeof(float));
    cudaMalloc(&d_Q, Q_size * Q_size * sizeof(float));
    cudaMemcpy(d_Q, Q.data(), Q_size * Q_size * sizeof(float), cudaMemcpyHostToDevice);
    // Initialize CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = numParticles, n = Q_size, k = Q_size;
    int lda = n, ldb = n, ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int t = 1; t <= t_max; ++t) 
    {
        // Perform the matrix multiplication Y = QX
        // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
        // h_particlesPositions = X^T
        // In CPU: Y = Q x X = Q x h_particlesPositions^T
        // But in GPU: we actually get on host (Q^T x X^T) = (X x Q)^T = Y^T
        // Therefore we have to do in cuBLAS C = A^T x B -> Y^T = X^T x Q^T where C = Y^T, A = X = h_particlesPositions^T, B = Q^T
        // For cuBlas, that is we do X x Q = h_particlesPositions^T x Q on it, so h_particlesPositions need transpose
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_particlesPositions, lda, d_Q, ldb, &beta, d_Y, ldc);
        // Check for any errors launching the kernel
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed in cublasSgemm: %s\n", cudaGetErrorString(error));
        }
        // cudaMemcpy(h_Y, d_Y, Q_size * numParticles * sizeof(float), cudaMemcpyDeviceToHost);

        // Update personal best
        computeFsUpdatePersonalBest<<<blocksPerGrid, threadsPerBlock>>>(
            d_particlesPositions,
            d_personalBestValues,
            d_personalBestPositions,
            d_Y,
            Q_size, 
            numParticles
        );
        cudaDeviceSynchronize();
        // Check for any errors launching the kernel
        error = cudaGetLastError();
        if (error != cudaSuccess) 
        {
            fprintf(stderr, "Kernel launch failed in computeFsUpdatePersonalBest: %s\n", cudaGetErrorString(error));
        }

        // Copy the values back to host
        cudaMemcpy(h_personalBestValues, d_personalBestValues, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_personalBestPositions, d_personalBestPositions, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);


        // Build the archive from non-dominated personal bests relative to the first particle's personal best
        std::vector<float> archivePersonalBestPositions;
        std::vector<float> archivePersonalBestValues;
        int archiveSize = 0;
        // Get values of first particle
        float f1_1 = h_personalBestValues[0], f2_1 = h_personalBestValues[1];
        for (int idx = 0; idx < numParticles; ++idx) 
        {
            // Get target values
            float personalBestf1 = h_personalBestValues[idx * 2], personalBestf2 = h_personalBestValues[idx * 2 + 1];
            // Skip if dominated by particle 1
            if ((f1_1 < personalBestf1 || f2_1 < personalBestf2) && (f1_1 <= personalBestf1 && f2_1 <= personalBestf2)) continue;
            archiveSize++;
            archivePersonalBestValues.push_back(personalBestf1);
            archivePersonalBestValues.push_back(personalBestf2);
            for (int j = 0; j < Q_size; ++j) archivePersonalBestPositions.push_back(h_personalBestPositions[idx * Q_size + j]);
        }
        // std::cout << "Archive size: " << archiveSize << std::endl;
        // Copy the values to device
        float* d_archivePersonalBestPositions;
        float* d_archivePersonalBestValues;
        cudaMalloc(&d_archivePersonalBestPositions, archiveSize * Q_size * sizeof(float));
        cudaMalloc(&d_archivePersonalBestValues, archiveSize * 2 * sizeof(float));
        cudaMemcpy(d_archivePersonalBestPositions, archivePersonalBestPositions.data(), archiveSize * Q_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_archivePersonalBestValues, archivePersonalBestValues.data(), archiveSize * 2 * sizeof(float), cudaMemcpyHostToDevice);



        // Update velocities and positions based on the archive
        updateVelocityAndPosition<<<blocksPerGrid, threadsPerBlock>>>(
        d_archivePersonalBestPositions,
        d_archivePersonalBestValues,
        d_particlesPositions,
        d_particlesVelocities,
        d_personalBestValues,
        d_personalBestPositions,
        d_globalBestPositions,
        d_globalBestValues,
        Q_size, 
        numParticles,
        archiveSize,
        w,
        c1, 
        c2);
        cudaDeviceSynchronize();
        // Check for any errors launching the kernel
        error = cudaGetLastError();
        if (error != cudaSuccess) 
        {
            fprintf(stderr, "Kernel launch failed in updateVelocityAndPosition: %s\n", cudaGetErrorString(error));
        }

    }


    // Check Matrix Y
    // std::cout << "Matrix Y:" << std::endl;
    // for (int i = 0; i < Q_size; ++i) {
    //     for (int j = 0; j < numParticles; ++j) {
    //     std::cout << h_Y[i * numParticles + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Copy the values back to host
    cudaMemcpy(h_personalBestValues, d_personalBestValues, numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_personalBestPositions, d_personalBestPositions, numParticles * Q_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Build the pareto archive
    // Build the archive from non-dominated personal bests relative to the first particle's personal best
    std::vector<float> h_archiveFinalPersonalBestPositions;
    int archiveFinalSize = 0;
    // Get values of first particle
    float f1_1 = h_personalBestValues[0], f2_1 = h_personalBestValues[1];
    for (int idx = 0; idx < numParticles; ++idx) 
    {
        // Get target values
        float personalBestf1 = h_personalBestValues[idx * 2], personalBestf2 = h_personalBestValues[idx * 2 + 1];
        // Skip if dominated by particle 1 (f1_1, f2_1)
        if (f1_1 <= personalBestf1 && f2_1 <= personalBestf2 && (f1_1 < personalBestf1 || f2_1 < personalBestf2)) continue;        archiveFinalSize++;
        for (int j = 0; j < Q_size; ++j) {
            h_archiveFinalPersonalBestPositions.push_back(std::round(h_personalBestPositions[idx * Q_size + j]));
        }
    }
    // Copy h_personalBestPositions to device
    float* d_archiveFinalPersonalBestPositions;
    cudaMalloc(&d_archiveFinalPersonalBestPositions, archiveFinalSize * Q_size * sizeof(float));
    cudaMemcpy(d_archiveFinalPersonalBestPositions, h_archiveFinalPersonalBestPositions.data(), archiveFinalSize * Q_size * sizeof(float), cudaMemcpyHostToDevice);
    // Calculate Y
    m = archiveFinalSize, n = Q_size, k = Q_size;
    // Find the one with minimal f1
    float* h_YFinal = new float[Q_size * archiveFinalSize];
    float* d_YFinal;
    cudaMalloc(&d_YFinal, Q_size * archiveFinalSize * sizeof(float));
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_archiveFinalPersonalBestPositions, lda, d_Q, ldb, &beta, d_YFinal, ldc);
    // Check for any errors launching the kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed in cublasSgemm: %s\n", cudaGetErrorString(error));
    }
    cudaMemcpy(h_YFinal, d_YFinal, Q_size * archiveFinalSize * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "The final archive size is: " <<archiveFinalSize << std::endl;
    // for (int i = 0; i < Q_size * archiveFinalSize; ++i) std::cout << h_YFinal[i] << " ";
    //  Find the optimal from the archive
    float minf1 = 0.0f;
    // int optimalIdx = 0;
    for (int i = 0; i < archiveFinalSize; ++i) 
    {
        float f1 = 0.0f;
        for (int j = 0; j < Q_size; j++) 
        {
            float xj = h_archiveFinalPersonalBestPositions[i * Q_size + j];
            f1 += xj * h_YFinal[j * archiveFinalSize + i];
        }
        // Check minimum
        if (minf1 > f1)
        {
            minf1 = f1;
            // optimalIdx = i;
            // std::cout << "min f1: " << minf1 << std::endl;
        }
    }

    // Stop recording
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GPU execution time: " << duration / 1000000.0f << " seconds\n";
    std::cout << "Best f1 value found:" << minf1 << std::endl;

    // Calculate relative error for accuracy of found solutions
    // float relativeError = (minf1 - fOptimal) / std::abs(fOptimal);

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
    cudaFree(d_archiveFinalPersonalBestPositions);
    cudaFree(d_YFinal);

    // Free host memory
    delete[] h_particlesPositions;
    delete[] h_particlesVelocities;
    delete[] h_personalBestPositions;
    delete[] h_personalBestValues;
    delete[] h_globalBestPositions;
    delete[] h_globalBestValues;
    delete[] h_YFinal;

}