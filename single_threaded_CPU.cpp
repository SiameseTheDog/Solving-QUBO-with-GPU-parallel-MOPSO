#include <vector>
#include <ctime>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>


// Particle structure representing a potential solution
struct Particle 
{
    std::vector<float> position;          // Real-valued position vector
    std::vector<float> velocity;          // Real-valued velocity vector
    std::vector<float> personalBestValues; // Stores f1 and f2 values
    std::vector<float> personalBestPosition; // Best personal position encountered by this particle
};

// GlobalBest structure representing the best solution found by the swarm
struct GlobalBest 
{
    std::vector<float> value;              // Tracking both f1 and f2
    std::vector<float> position;          // Real-valued position vector representing the best global solution
};

// Initialize a particle with random positions and zero velocities
void initializeParticle(Particle& p, size_t dim) 
{
    p.position.resize(dim);
    p.velocity.resize(dim);
    p.personalBestPosition.resize(dim);

    for (size_t i = 0; i < dim; i++) 
    {
        p.position[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // 0 <= 𝑥 <= 1
        p.velocity[i] = 0.0f; // Initial velocity is zero
        p.personalBestPosition[i] = p.position[i];
    }

    p.personalBestValues.assign(2, std::numeric_limits<float>::infinity()); // Initialize with 'worst' values

}

// Function to combine particle positions into matrix X
std::vector<std::vector<float>> combineParticlePositions(const std::vector<Particle>& particles, int Q_size, int numParticles) {
    std::vector<std::vector<float>> X(Q_size, std::vector<float>(numParticles, 0.0f));
    for (size_t j = 0; j < numParticles; ++j) {
        for (int i = 0; i < Q_size; ++i) {
            X[i][j] = particles[j].position[i];
        }
    }
    return X;
}

// Function to perform matrix-matrix multiplication (QX)
std::vector<std::vector<float>> multiplyQX(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<float>>& X, int Q_size, int numParticles) {
    std::vector<std::vector<float>> Y(Q_size, std::vector<float>(numParticles, 0.0f));
    for (size_t i = 0; i < Q_size; ++i) {
        for (size_t j = 0; j < numParticles; ++j) {
            for (size_t k = 0; k < Q_size; ++k) {
                Y[i][j] += Q[i][k] * X[k][j];
            }
        }
    }
    return Y;
}
// Overload for const std::vector<std::vector<int>>& X
std::vector<std::vector<float>> multiplyQX(const std::vector<std::vector<float>>& Q, const std::vector<std::vector<int>>& X, int Q_size, int numParticles) {
    std::vector<std::vector<float>> Y(Q_size, std::vector<float>(numParticles, 0.0f));
    for (size_t i = 0; i < Q_size; ++i) {
        for (size_t j = 0; j < numParticles; ++j) {
            for (size_t k = 0; k < Q_size; ++k) {
                Y[i][j] += Q[i][k] * X[k][j];
            }
        }
    }
    return Y;
}

// Objective function f1 calculation (x'Qx = inner product of ith column of Y with ith column of X)
float f1(const std::vector<std::vector<float>>& Y, const std::vector<std::vector<float>>& X, size_t particleIndex) 
{
    float result = 0.0f;
    // Assuming that Y and X have the same number of rows as the dimension of the problem
    // and the same number of columns as the number of particles.
    for (size_t i = 0; i < X.size(); i++) { // X.size() should be equal to the dimension of the problem
        result += Y[i][particleIndex] * X[i][particleIndex];
    }
    return result;
}
// Overload for std::vector<int>
float f1(const std::vector<std::vector<float>>& Y, const std::vector<std::vector<int>>& X, size_t particleIndex) 
{
    float result = 0.0f;
    // Assuming that Y and X have the same number of rows as the dimension of the problem
    // and the same number of columns as the number of particles.
    for (size_t i = 0; i < X.size(); i++) { // X.size() should be equal to the dimension of the problem
        result += Y[i][particleIndex] * X[i][particleIndex];
    }
    return result;
}

// Objective function f2 calculation (Sum(xi * (1 - xi)))
float f2(const std::vector<float>& x) 
{
    float result = 0.0f;
    for (float xi : x) result += xi * (1.0f - xi);
    return result;
}

// Update the velocity and position of the particle according to PSO rules
void updateParticle(Particle& p, const std::vector<float>& gBestPosition, float w, float c1, float c2, int dim) {
    // Velocity and position update based on PSO equations
    for (int i = 0; i < dim; i++) {
        float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for cognitive component
        float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random factor for social component
        
        // Update velocity for each dimension
        p.velocity[i] = w * p.velocity[i] +                                      
                        c1 * r1 * (p.personalBestPosition[i] - p.position[i]) +  
                        c2 * r2 * (gBestPosition[i] - p.position[i]);            

        // Update position for each dimension
        p.position[i] += p.velocity[i];
        
         // Clamping to ensure position is within [0,1]
        if (p.position[i] < 0.0) p.position[i] = 0.0;
        if (p.position[i] > 1.0) p.position[i] = 1.0;
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

int main() 
{
    // Initialize random seed
    srand(static_cast<unsigned>(time(nullptr)));

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
    std::vector<std::vector<float>> Q = 
    {
        {-526, 150, 160, 190, 180,  20,  40, -30, -60, -120},
        { 150, -574, 180, 200, 200,  20,  40, -30, -60, -120},
        { 160, 180, -688, 220, 200,  40,  80, -20, -40, -80},
        { 190, 200, 220, -645, 240,  30,  60, -40, -80, -160},
        { 180, 200, 200, 240, -605,  20,  40, -40, -80, -160},
        {  20,  20,  40,  30,  20, -130,  20,   0,   0,    0},
        {  40,  40,  80,  60,  40,  20, -240,   0,   0,    0},
        { -30, -30, -20, -40, -40,   0,   0, 110,  20,   40},
        { -60, -60, -40, -80, -80,   0,   0,  20, 240,   80},
        { -120, -120, -80, -160, -160,   0,   0,  40,  80, 560}
    };
    int Q_size = Q.size();
    // const float fOptimal = -6889.0f; // Known optimal objective value for comparison
    const float fOptimal = -916.0f;

    // Initialize particles and global best
    std::vector<Particle> particles(numParticles);
    for (Particle& p : particles) initializeParticle(p, Q_size);
    
    GlobalBest globalBest;
    globalBest.value.assign(2, std::numeric_limits<float>::infinity()); // Initialize with 'worst' values
    globalBest.position.resize(Q_size, 0.0f);

    // Main optimization loop as described in Algorithm 1
    for (int t = 1; t <= t_max; t++) 
    {
        // Update particle with the PSO formula
        for (auto& p : particles) updateParticle(p, globalBest.position, w, c1, c2, Q_size);
        
        // Combine particle positions into a single matrix X
        std::vector<std::vector<float>> X = combineParticlePositions(particles, Q_size, numParticles);
        // Perform the matrix multiplication QX
        std::vector<std::vector<float>> Y = multiplyQX(Q, X, Q_size, numParticles);

        for (size_t particleIndex = 0; particleIndex < particles.size(); ++particleIndex)
        {
            auto& p = particles[particleIndex];

            // Calculate objective function values
            std::vector<float> objectives = {f1(Y, X, particleIndex), f2(p.position)};

            if (dominates(objectives, p.personalBestValues)) 
            {
                p.personalBestPosition = p.position;
                p.personalBestValues = objectives;
            }
        }

        // Build the archive from non-dominated personal bests relative to the first particle's personal best
        std::vector<Particle> archive;
        for (const auto& p : particles) 
        {
            if (!dominates(particles[0].personalBestValues, p.personalBestValues)) archive.push_back(p);
        }

        // Update velocities and positions based on the archive
        for (auto& p : particles) 
        {
            if (!archive.empty()) 
            {
                int randomIndex = rand() % archive.size();
                const auto& globalBest = archive[randomIndex];
                updateParticle(p, globalBest.personalBestPosition, w, c1, c2, Q_size);
            }
        }
    }

    std::vector<std::vector<int>> roundedSolutions(Q_size, std::vector<int>(numParticles, 0));
    // Round solutions and get Y
    for (size_t particleIndex = 0; particleIndex < numParticles; ++particleIndex) 
    {
        auto& p = particles[particleIndex];
        for (int i = 0; i < Q_size; ++i) 
        {
            roundedSolutions[i][particleIndex] = std::round(p.position[i]);
        }
    }
    std::vector<std::vector<float>> Y_rounded = multiplyQX(Q, roundedSolutions, Q_size, numParticles);

    // Find the one with minimal f1
    float minimalF1 = std::numeric_limits<float>::max();
    std::vector<int> bestRoundedSolution(Q_size); // This will store the best rounded solution


    for (size_t particleIndex = 0; particleIndex < numParticles; ++particleIndex)
    {
        float currentF1 = f1(Y_rounded, roundedSolutions, particleIndex);
        if (currentF1 < minimalF1) 
        {
            minimalF1 = currentF1;
            // Update bestRoundedSolution with integer values
            bestRoundedSolution = roundedSolutions[particleIndex];
        }
    }

    // Calculate relative error for accuracy of found solutions
    float relativeError = (minimalF1 - fOptimal) / std::abs(fOptimal);

    // Output the best rounded solution and its f1 value
    std::cout << "Best solution found:" << std::endl;
    for (int val : bestRoundedSolution) std::cout << val << " ";
    std::cout << "\nRelative error compared to real optimal f1: " << relativeError * 100 << "%" << std::endl;
}

