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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path> <num_particles>" << std::endl;
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

    int numParticles = std::stoi(argv[2]);

    // Define parameters as in the paper
    const float w = 0.729;   // Inertia weight
    const float c1 = 1.4955; // Cognitive weight
    const float c2 = 1.4955; // Social weight
    const int t_max = 1; // Maximum number of iterations

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize random seed
    unsigned int seed = static_cast<unsigned>(time(nullptr));
    srand(seed);

    /************************************ Position Initialization *********************************/
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
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed in initializeParticleKernel: %s\n", cudaGetErrorString(error));
    }

    /******************************** End of Position Initialization ********************************/

    // Remaining code for PSO algorithm...

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GPU execution time: " << duration / 1000000.0f << " seconds\n";
    std::cout << "Best f1 value found:" << minf1 << std::endl;

    // Clean up
    cudaFree(d_particlesPositions);
    cudaFree(d_particlesVelocities);
    cudaFree(d_personalBestValues);
    cudaFree(d_personalBestPositions);
    cudaFree(d_globalBestPositions);
    cudaFree(d_globalBestValues);

    delete[] h_particlesPositions;
    delete[] h_particlesVelocities;
    delete[] h_personalBestValues;
    delete[] h_personalBestPositions;
    delete[] h_globalBestPositions;
    delete[] h_globalBestValues;

    return 0;
}
