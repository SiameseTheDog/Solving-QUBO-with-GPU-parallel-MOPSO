#include <vector>
#include <ctime>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <cblas.h>
#include <chrono>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


// Initialize particles with random positions and zero velocities
void initializeParticles(int numParticles, int Q_size, std::vector<float>& positions, std::vector<float>& velocities, std::vector<float>& personalBestPositions, std::vector<float>& personalBestValues) {
    for (int i = 0; i < numParticles; ++i) {
        for (int j = 0; j < Q_size; ++j) {
            int positionIdx = i * Q_size + j;
            positions[positionIdx] = static_cast<float>(rand()) / RAND_MAX; // Initialize positions
            velocities[positionIdx] = 0.0f; // Initialize velocities
            personalBestPositions[positionIdx] = positions[positionIdx]; // Copy initial position
        }
        // Initialize personal best values to a large number
        personalBestValues[i * 2] = std::numeric_limits<float>::infinity();
        personalBestValues[i * 2 + 1] = std::numeric_limits<float>::infinity();
    }
}

// Function to check if solution 'a' dominates solution 'b'
bool dominates(const std::vector<float>& a, const std::vector<float>& b) 
{
    bool anyBetter = false;
    for (int i = 0; i < 2; ++i) 
    {
        if (a[i] > b[i]) return false; // 'a' can't be worse in any objective
        if (a[i] < b[i]) anyBetter = true; // 'a' has to be better in at least one objective
    }
    return anyBetter;
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
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path> <num_particles>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    int numParticles = std::stoi(argv[2]);  // Convert the second argument to integer for number of particles

    std::vector<float> Q;
    int Q_size = 0;

    if (!readMatrix(filePath, Q, Q_size)) {
        std::cerr << "Error reading matrix from file: " << filePath << std::endl;
        return 1;
    }

    // std::cout << "Q_size: " << Q_size << std::endl;
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
    const float c1 = 1.4595; // Cognitive weight
    const float c2 = 1.4595; // Social weight
    const int t_max = 2500; // Maximum number of iterations
    
    // // Define the Q matrix, its size and result value of real solution
    // // Examples are from page 9 of 'A Tutorial on Formulating and Using QUBO Models'
    
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
    // int Q_size = 8;
    // const float fOptimal = -6889.0f; // Known optimal objective value for comparison


    // // exmple from page 34
    // std::vector<float> Q = {
    //     -526, 150, 160, 190, 180,  20,  40, -30, -60, -120,
    //     150, -574, 180, 200, 200,  20,  40, -30, -60, -120,
    //     160, 180, -688, 220, 200,  40,  80, -20, -40, -80,
    //     190, 200, 220, -645, 240,  30,  60, -40, -80, -160,
    //     180, 200, 200, 240, -605,  20,  40, -40, -80, -160,
    //     20,  20,  40,  30,  20, -130,  20,   0,   0,    0,
    //     40,  40,  80,  60,  40,  20, -240,   0,   0,    0,
    //     -30, -30, -20, -40, -40,   0,   0, 110,  20,   40,
    //     -60, -60, -40, -80, -80,   0,   0,  20, 240,   80,
    //     -120, -120, -80, -160, -160,   0,   0,  40,  80, 560
    // };
    // int Q_size = 10;
    // const float fOptimal = -916.0f;

    // Create separate vectors for each component
    // Note: All are flattened in row-major way
    std::vector<float> positions(numParticles *Q_size, 0.0f);
    std::vector<float> velocities(numParticles * Q_size, 0.0f);
    std::vector<float> personalBestPositions(numParticles * Q_size, 0.0f);
    std::vector<float> personalBestValues(numParticles * 2, std::numeric_limits<float>::infinity());

    auto startMultiThreaded  = std::chrono::high_resolution_clock::now();

    int numThreads = std::thread::hardware_concurrency();
    openblas_set_num_threads(numThreads);

    // Initialize particles and global best
    initializeParticles(numParticles, Q_size, positions, velocities, personalBestPositions, personalBestValues);
    // // Print positions to verify initialization
    // for (int i = 0; i < numParticles; ++i) {
    //     std::cout << "Particle " << i << " positions: ";
    //     for (int j = 0; j < Q_size; ++j) {
    //         std::cout << positions[i * Q_size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // Test positions for matrix 1;
        // positions ={0,0,0,1,1,0,0,1,
        //             0,0,0,1,1,0,0,1,
        //             0,0,0,1,1,0,0,1};

    // Main optimization loop as described in Algorithm 1
    std::vector<float> Y(Q_size * numParticles, 0.0f);
    for (int t = 1; t <= t_max; ++t) 
    {
        // std::cout << "Iteration " << t << std::endl;
        // Perform the matrix multiplication Y = Q x X = Q x positions^T
        // Perform the matrix multiplication using CBLAS
        // https://www.ibm.com/docs/en/essl/6.3?topic=mos-sgemm-dgemm-cgemm-zgemm-combined-matrix-multiplication-addition-general-matrices-their-transposes-conjugate-transposes

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                Q_size, numParticles, Q_size, // dimensions of matrices
                1.0, // alpha
                Q.data(), Q_size, // A and its leading dimension
                positions.data(), Q_size, // B and its leading dimension
                0.0, // beta
                Y.data(), numParticles); // C and its leading dimension

        for (size_t particleIndex = 0; particleIndex < numParticles; ++particleIndex)
        {
                        // Calculate objective function values
            float f1 = 0.0f, f2 = 0.0f;
            for (int j = 0; j < Q_size; j++) 
            {
                float xj = positions[particleIndex * Q_size + j];
                f1 += xj * Y[j * numParticles + particleIndex];
                f2 += xj * (1.0f - xj);
            }
            // std::cout << "idx: " << particleIndex << " " << f1 << std::endl;
            std::vector<float> objectives = {f1, f2};
            std::vector<float> personalBest = {personalBestValues[particleIndex * 2], personalBestValues[particleIndex * 2 + 1]};
   
            // Update person best values and positions
            if (dominates(objectives, personalBest)) 
            {
                personalBestValues[particleIndex * 2] = f1;
                personalBestValues[particleIndex * 2 + 1] = f2;
                for (int j = 0; j < Q_size; ++j) personalBestPositions[particleIndex * Q_size + j] = positions[particleIndex * Q_size + j];
                // std::cout << "in " << t << "round, particle index "<< particleIndex << " dominate " << personalBest[0] << " " << personalBest[1] << " with " << f1 <<" " << f2 <<std::endl;
            }
        }

        // Build the archive from non-dominated personal bests relative to the first particle's personal best
        std::vector<float> archivePersonalBestPositions;
        std::vector<float> archivePersonalBestValues;
        int archiveSize = 0;
        std::vector<float> particle0personalBest = {personalBestValues[0], personalBestValues[1]};
        for (int particleIndex = 0; particleIndex < numParticles; ++particleIndex) 
        {
            std::vector<float> personalBest = {personalBestValues[particleIndex * 2], personalBestValues[particleIndex * 2 + 1]};
            if (!dominates(particle0personalBest, personalBest))
            {
                archiveSize++;
                archivePersonalBestValues.push_back(personalBest[0]);
                archivePersonalBestValues.push_back(personalBest[1]);
                for (int j = 0; j < Q_size; ++j) archivePersonalBestPositions.push_back(personalBestPositions[particleIndex * Q_size + j]);            }
        }
        // std::cout << "Archive size is: " << archiveSize << std::endl;

        // Update velocities and positions based on the archive and PSO formula
        for (int particleIndex = 0; particleIndex < numParticles; ++particleIndex) 
        {
            int randomIndex = rand() % archiveSize;
            float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for cognitive component
            float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for social component
            for (int j = 0; j < Q_size; j++) 
            {
                int positionIdx = particleIndex * Q_size + j;
                //  Update speed
                int globalBestPositionIdx = randomIndex * Q_size + j;
                velocities[positionIdx] = w * velocities[positionIdx] +                                      
                                          c1 * r1 * (personalBestPositions[positionIdx] - positions[positionIdx]) +  
                                          c2 * r2 * (archivePersonalBestPositions[globalBestPositionIdx] - positions[positionIdx]);            
                // Update position
                positions[positionIdx] += velocities[positionIdx];
                // Clamping to ensure position is within [0,1]
                if (positions[positionIdx] < 0.0) positions[positionIdx] = 0.0;
                if (positions[positionIdx] > 1.0) positions[positionIdx] = 1.0;
            }
        }
    }
    // // Print positions to verify main loop
    // for (int i = 0; i < numParticles; ++i) 
    // {
    //     std::cout << "Particle " << i << " positions: ";
    //     for (int j = 0; j < Q_size; ++j) {
    //         std::cout << positions[i * Q_size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // Build the archive from non-dominated personal bests relative to the first particle's personal best
    std::vector<float> finalRoundedPositions;
    int finalArchiveSize = 0;
    std::vector<float> particle0personalBest = {personalBestValues[0], personalBestValues[1]};
    for (int particleIndex = 0; particleIndex < numParticles; ++particleIndex) 
    {
            std::vector<float> personalBest = {personalBestValues[particleIndex * 2], personalBestValues[particleIndex * 2 + 1]};
            if (!dominates(particle0personalBest, personalBest))
            {
                finalArchiveSize++;
                // Round and record the solutions
                for (int j = 0; j < Q_size; ++j) finalRoundedPositions.push_back(std::round(personalBestPositions[particleIndex * Q_size + j]));            
            
            }
    }

    // Get Y
    std::vector<float> Y_fianl(Q_size * numParticles);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            Q_size, finalArchiveSize, Q_size, // dimensions of matrices
            1.0, // alpha
            Q.data(), Q_size, // A and its leading dimension
            finalRoundedPositions.data(), Q_size, // B and its leading dimension
            0.0, // beta
            Y_fianl.data(), finalArchiveSize); // C and its leading dimension
    // for (size_t particleIndex = 0; particleIndex < finalArchiveSize; ++particleIndex) 
    //     {
    //         for (int i = 0; i < Q_size; ++i) 
    //         {
    //             std::cout << Y_rounded[i][particleIndex] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
        
    // Find the one with minimal f1
    float minimalF1 = std::numeric_limits<float>::max();
    int bestIdx = 0;
    for (size_t particleIndex = 0; particleIndex < finalArchiveSize; ++particleIndex)
    {
        float currentF1 = 0;
        for (int j = 0; j < Q_size; j++) 
        {
            float xj = finalRoundedPositions[particleIndex * Q_size + j];
            currentF1 += xj * Y_fianl[j * finalArchiveSize + particleIndex];
        }
        if (currentF1 < minimalF1) 
        {
            minimalF1 = currentF1;
            // // Update bestRoundedSolution idx with integer values
            bestIdx = particleIndex;
        }
    }

    auto endMultiThreaded = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedMultiThreaded = endMultiThreaded - startMultiThreaded;
    std::cout << "Multi-threaded execution time for " << numParticles <<" particles: " << elapsedMultiThreaded.count() << " seconds\n";

    // Output the best rounded solution and its f1 value
    std::cout << "Best solution found for" << filePath << ": "<< " " << minimalF1 << std::endl << std::endl;
    // for (int j = 0; j < Q_size; j++) 
    // {
    //     std::cout << finalRoundedPositions[bestIdx * Q_size + j] << " ";
    // }
    
}