#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Domain and grid parameters
const double Lx = 10.0, Ly = 10.0, Lz = 10.0; // Domain size
const int Nx = 100, Ny = 100, Nz = 100;       // Grid resolution
const double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz; // Grid spacing
const double CFL = 0.5;                       // CFL condition constant
const double t_final = 10.0;                  // Simulation end time
const int max_iterations = 500;              // Max iterations for Poisson solver

// Physical constants
const double mu0 = 1.0;                       // Magnetic permeability
const double q_e = -1.6e-19, m_e = 9.1e-31;   // Electron charge, mass
const double c = 3.0e8;                       // Speed of light
const double epsilon = 1e-6;                  // Small value to avoid division by zero

// Variables: density, velocity, magnetic field, electric field, potential
std::vector<std::vector<std::vector<double>>> rho(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 1.0)));
std::vector<std::vector<std::vector<double>>> vx(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> vy(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> vz(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Bx(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> By(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Bz(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 1.0)));
std::vector<std::vector<std::vector<double>>> Ex(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Ey(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Ez(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Jx(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Jy(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> Jz(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0)));
std::vector<std::vector<std::vector<double>>> phi(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nz, 0.0))); // Poisson potential

// Particle structure
struct Particle {
    double x, y, z;      // Position
    double vx, vy, vz;   // Velocity
    double charge, mass; // Charge and mass
};
std::vector<Particle> particles;

// Function prototypes
double compute_time_step();
void initialize_conditions();
void initialize_particles(int num_particles);
void particle_push(double dt);
void apply_boundary_conditions();
void compute_hall_term();
void current_deposition();
double compute_divergence_B();
void clean_divergence_B();
void solve();

int main() {
    initialize_conditions();
    initialize_particles(1000); // Initialize 1000 particles
    solve();
    return 0;
}

// Function to compute adaptive time step
double compute_time_step() {
    double max_speed = c; // Include speed of light for stability
    for (const auto &p : particles) {
        double speed = std::sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
        max_speed = std::max(max_speed, speed);
    }
    return CFL * std::min({dx, dy, dz}) / max_speed;
}

// Function to apply initial conditions
void initialize_conditions() {
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i * dx - Lx / 2.0;
                double y = j * dy - Ly / 2.0;
                double z = k * dz - Lz / 2.0;

                // Harris sheet with perturbation
                rho[i][j][k] = 1.0 + 0.1 * std::exp(-y * y / 0.5);
                Bz[i][j][k] = std::tanh(y / 0.5);
                vx[i][j][k] = 0.1 * std::sin(2 * M_PI * z / Lz);
                Bx[i][j][k] = 0.01 * std::cos(2 * M_PI * x / Lx);
            }
        }
    }
}

// Initialize particles using std::uniform_real_distribution
void initialize_particles(int num_particles) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_x(0.0, Lx);
    std::uniform_real_distribution<> dist_y(0.0, Ly);
    std::uniform_real_distribution<> dist_z(0.0, Lz);

    for (int i = 0; i < num_particles; ++i) {
        Particle p = {dist_x(gen), dist_y(gen), dist_z(gen),
                      0.0, 0.0, 0.0, 1.0, 1.0};
        particles.push_back(p);
    }
}

// Function to compute Hall term (Placeholder)
void compute_hall_term() {
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                Ex[i][j][k] += (Jy[i][j][k] * Bz[i][j][k] - Jz[i][j][k] * By[i][j][k]) / rho[i][j][k];
                Ey[i][j][k] += (Jz[i][j][k] * Bx[i][j][k] - Jx[i][j][k] * Bz[i][j][k]) / rho[i][j][k];
                Ez[i][j][k] += (Jx[i][j][k] * By[i][j][k] - Jy[i][j][k] * Bx[i][j][k]) / rho[i][j][k];
            }
        }
    }
}

// Function to deposit current (Placeholder)
void current_deposition() {
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                Jx[i][j][k] = Jy[i][j][k] = Jz[i][j][k] = 0.0;
            }
        }
    }

    for (const auto &p : particles) {
        int i = std::floor(p.x / dx), j = std::floor(p.y / dy), k = std::floor(p.z / dz);
        Jx[i][j][k] += p.charge * p.vx;
        Jy[i][j][k] += p.charge * p.vy;
        Jz[i][j][k] += p.charge * p.vz;
    }
}

// Function to update particle positions and velocities
void particle_push(double dt) {
    for (auto &p : particles) {
        int i = std::floor(p.x / dx), j = std::floor(p.y / dy), k = std::floor(p.z / dz);
        double wx = (p.x - i * dx) / dx;
        double wy = (p.y - j * dy) / dy;
        double wz = (p.z - k * dz) / dz;

        // Interpolate fields
        double Ex_p = Ex[i][j][k];
        double Ey_p = Ey[i][j][k];
        double Ez_p = Ez[i][j][k];
        double Bx_p = Bx[i][j][k];
        double By_p = By[i][j][k];
        double Bz_p = Bz[i][j][k];

        // Boris Particle Push Algorithm
        double vxm = p.vx + (p.charge / p.mass) * Ex_p * dt / 2.0;
        double vym = p.vy + (p.charge / p.mass) * Ey_p * dt / 2.0;
        double vzm = p.vz + (p.charge / p.mass) * Ez_p * dt / 2.0;

        double t_x = (p.charge / p.mass) * Bx_p * dt / 2.0;
        double t_y = (p.charge / p.mass) * By_p * dt / 2.0;
        double t_z = (p.charge / p.mass) * Bz_p * dt / 2.0;

        double t_mag = t_x * t_x + t_y * t_y + t_z * t_z;
        double s_factor = 2.0 / (1.0 + t_mag);

        double vtx = vxm + vym * t_z - vzm * t_y;
        double vty = vym + vzm * t_x - vxm * t_z;
        double vtz = vzm + vxm * t_y - vym * t_x;

        double vxp = vtx + s_factor * (vty * t_z - vtz * t_y);
        double vyp = vty + s_factor * (vtz * t_x - vtx * t_z);
        double vzp = vtz + s_factor * (vtx * t_y - vty * t_x);

        p.vx = vxp + (p.charge / p.mass) * Ex_p * dt / 2.0;
        p.vy = vyp + (p.charge / p.mass) * Ey_p * dt / 2.0;
        p.vz = vzp + (p.charge / p.mass) * Ez_p * dt / 2.0;

        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;

        // Apply periodic boundary conditions
        p.x = fmod(p.x + Lx, Lx);
        p.y = fmod(p.y + Ly, Ly);
        p.z = fmod(p.z + Lz, Lz);
    }
}

// Function to compute divergence of B
double compute_divergence_B() {
    double max_divB = 0.0;
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                double divB = (Bx[i + 1][j][k] - Bx[i - 1][j][k]) / (2 * dx) +
                              (By[i][j + 1][k] - By[i][j - 1][k]) / (2 * dy) +
                              (Bz[i][j][k + 1] - Bz[i][j][k - 1]) / (2 * dz);
                max_divB = std::max(max_divB, std::abs(divB));
            }
        }
    }
    return max_divB;
}

// Function to apply boundary conditions
void apply_boundary_conditions() {
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
                    Bx[i][j][k] *= 0.95; // Absorbing boundary
                    By[i][j][k] *= 0.95;
                    Bz[i][j][k] *= 0.95;
                }
            }
        }
    }
}

// Function to clean divergence of B
void clean_divergence_B() {
    double initial_divB = compute_divergence_B();

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                for (int k = 1; k < Nz - 1; ++k) {
                    phi[i][j][k] = (phi[i + 1][j][k] + phi[i - 1][j][k] +
                                    phi[i][j + 1][k] + phi[i][j - 1][k] +
                                    phi[i][j][k + 1] + phi[i][j][k - 1] - initial_divB) / 6.0;
                }
            }
        }
    }

    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                Bx[i][j][k] -= phi[i][j][k] / dx;
                By[i][j][k] -= phi[i][j][k] / dy;
                Bz[i][j][k] -= phi[i][j][k] / dz;
            }
        }
    }
}

// Main solver
void solve() {
    double t = 0.0;

    while (t < t_final) {
        double dt = compute_time_step();

        compute_hall_term();
        current_deposition();
        clean_divergence_B();
        particle_push(dt);
        apply_boundary_conditions();

        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                for (int k = 1; k < Nz - 1; ++k) {
                    Bx[i][j][k] += dt * ((Ey[i][j][k + 1] - Ey[i][j][k - 1]) / (2 * dz) -
                                         (Ez[i][j + 1][k] - Ez[i][j - 1][k]) / (2 * dy));

                    By[i][j][k] += dt * ((Ez[i + 1][j][k] - Ez[i - 1][j][k]) / (2 * dx) -
                                         (Ex[i][j][k + 1] - Ex[i][j][k - 1]) / (2 * dz));

                    Bz[i][j][k] += dt * ((Ex[i][j + 1][k] - Ex[i][j - 1][k]) / (2 * dy) -
                                         (Ey[i + 1][j][k] - Ey[i - 1][j][k]) / (2 * dx));
                }
            }
        }

        t += dt;

        double max_divB = compute_divergence_B();
        if (std::isinf(max_divB) || std::isnan(max_divB)) {
            std::cerr << "Numerical instability detected! Stopping simulation.\n";
            break;
        }

        std::cout << "Time: " << t << " / " << t_final
                  << ", Max divergence of B: " << max_divB << std::endl;
    }

    std::cout << "Simulation complete.\n";
}
