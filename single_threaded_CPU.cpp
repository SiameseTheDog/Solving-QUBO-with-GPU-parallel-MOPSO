#include <vector>
#include <ctime>
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

// Function to perform matrix-vector multiplication (Qx)
std::vector<float> multiplyQx(const std::vector<std::vector<float>>& Q, const std::vector<float>& x) {
    std::vector<float> Y(Q.size(), 0.0f);
    for (size_t i = 0; i < Q.size(); i++) 
    {
        for (size_t j = 0; j < Q[i].size(); j++) Y[i] += Q[i][j] * x[j];
    }
    return Y;
}

// Objective function f1 calculation (x'Qx)
float f1(const std::vector<float>& Y, const std::vector<float>& x) 
{
    float result = 0.0f;
    for (size_t i = 0; i < x.size(); i++) result += x[i] * Y[i];
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
    const int numParticles = 30; // Number of particles swarm, may vary

    // Define the Q matrix, its size and result value of real solution
    // Example is from 'A Tutorial on Formulating and Using QUBO Models'
    std::vector<std::vector<float>> Q = 
    {
        {-3525, 175, 325, 775, 1050, 425, 525, 250},
        {175, -1113, 91, 217, 294, 119, 147, 70},
        {325, 91, -1989, 403, 546, 221, 273, 130},
        {775, 217, 403, -4185, 1302, 527, 651, 310},
        {1050, 294, 546, 1302, -5208, 714, 882, 420},
        {425, 119, 221, 527, 714, -2533, 357, 170},
        {525, 147, 273, 651, 882, 357, -3045, 210},
        {250, 70, 130, 310, 420, 170, 210, -1560}
    };
    int Q_size = Q.size();
    const float fOptimal = -6889.0f; // Known optimal objective value for comparison

    // Initialize particles and global best
    std::vector<Particle> particles(numParticles);
    for (Particle& p : particles) initializeParticle(p, Q_size);
    
    GlobalBest globalBest;
    globalBest.value.assign(2, std::numeric_limits<float>::infinity()); // Initialize with 'worst' values
    globalBest.position.resize(Q_size, 0.0f);

    // Main optimization loop as described in Algorithm 1
    for (int t = 1; t <= t_max; t++) 
    {
        for (auto& p : particles) 
        {
            // Update particle with the PSO formula
            updateParticle(p, globalBest, w, c1, c2, Q_size);
            
            // Calculate objective function values
            std::vector<float> Y = multiplyQx(Q, p.position);
            std::vector<float> objectives = {f1(Y, p.position), f2(p.position)};

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
    
    std::vector<int> bestRoundedSolution(Q_size); // This will store the best solution after rounding
    float minimalF1 = std::numeric_limits<float>::max();

    // Round solutions and find the one with minimal f1
    for (const auto& p : particles) 
    {
        std::vector<float> roundedSolution(Q_size);

        for (int i = 0; i < Q_size; ++i) roundedSolution[i] = std::round(p.position[i]);
        std::vector<float> Y = multiplyQx(Q, roundedSolution);
        float currentF1 = f1(Y, roundedSolution);
        if (currentF1 < minimalF1) 
        {
            minimalF1 = currentF1;
            // Update bestRoundedSolution with integer values
            for (size_t i = 0; i < Q_size; ++i) bestRoundedSolution[i] = static_cast<int>(std::round(p.position[i]));
        }
    }

    // Calculate relative error for accuracy of found solutions
    float relativeError = (minimalF1 - optimalF1) / optimalF1;

    // Output the best rounded solution and its f1 value
    std::cout << "Best solution found:" << std::endl;
    for (int val : bestRoundedSolution) {
        std::cout << val << " ";
    }
    std::cout << "\nRelative error compared to real optimal f1: " << relativeError * 100 << "%" << std::endl;
}