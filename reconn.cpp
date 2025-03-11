#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <string>
#include <fstream>

// Declare WENOWeights structure
struct WENOWeights {
    double alpha0;
    double alpha1;
    double alpha2;
};

// Function prototypes
std::vector<WENOWeights> precomputeWENOWeightsMPI();

// Constants
const double MU_0 = 4 * M_PI * 1e-7; // Vacuum permeability (N/A^2)
const double QE = 1.6e-19;           // Elementary charge (C)
const double ME = 9.11e-31;          // Electron mass (kg)
const double EPS_0 = 8.85e-12;       // Vacuum permittivity (F/m)
const double B0 = 5.0e-9;            // Background magnetic field (T)
const double MI = 1.67e-27;          // Ion mass (kg)
const double QI = 1.6e-19;           // Ion charge (C)
const double GAMMA = 5.0 / 3.0;      // Adiabatic index for ions
const double k_B = 1.38e-23;         // Boltzmann constant (J/K)
const double TOTAL_SIMULATION_TIME = 100.0; // Simulation time (s)
const int DIAGNOSTIC_INTERVAL = 100.0;
const int MONITOR_INTERVAL = 100;  // Define globally or pass to the function

// Domain parameters
const int NX = 192;               // Grid size in x-direction
const int NY = 192;               // Grid size in y-direction
const double DX = 1.0;            // Grid spacing in x-direction (m)
const double DY = 1.0;            // Grid spacing in y-direction (m)
const double DT = 0.01;           // Time step (s)
const int NUM_TIMESTEPS = 1000;   // Number of timesteps
const int multigrid_levels = static_cast<int>(std::log2(std::min(NX, NY))) - 1;
int px = 4; // Number of processes in x-direction
int py = 4; // Number of processes in y-direction
int local_NX = NX / px; // Local grid size in x-direction
int local_NY = NY / py; // Local grid size in y-direction
const double injection_area = NX * DY;  // Area at the inflow boundary

// Data extracted from THEMIS and MMS
const double Egsm_x_1 = -103.52e-3; // mV/m -> V/m
const double Egsm_y_1 = -3.53e-3;   // mV/m -> V/m
const double density_1 = 6.675e6;   // cm^-3 -> m^-3

// Injection parameters(SOLAR Acttivity of March 17 2015)
const double MIN_SOLAR_WIND_DENSITY = 2.76e6;  // Minimum solar wind density (particles/m³)
const double MAX_SOLAR_WIND_DENSITY = 32.87e6; // Maximum solar wind density (particles/m³)
const double AVG_SOLAR_WIND_DENSITY = 13e6; // Average solar wind density (particles/m³)
const double MIN_SOLAR_WIND_SPEED = 396.5e3;   // Minimum solar wind speed (m/s)
const double MAX_SOLAR_WIND_SPEED = 640.0e3;   // Maximum solar wind speed (m/s)
const double AVG_SOLAR_WIND_SPEED = 525e3; // Average solar wind speed (m/s)

// Inflow
const double inflow_temperature = 264.31;      // K (from THEMIS ESA)
const double inflow_magnetic_field_x = -5.768013e-9; // T (from THEMIS FGM, converted from nT)
const double inflow_magnetic_field_y = 5.463308e-9;  // T (from THEMIS FGM, converted from nT)
const double inflow_magnetic_field_z = -12.34e-9; // T (from MMS MEC, converted from nT)
const double inflow_electron_temperature = 54090.38;  // K (from THEMIS SST)
const double inflow_electron_density = 1.77824677e-05 * 1e6; // particles/m^3 (from THEMIS SST, converted from cm^-3)
const double inflow_electron_velocity_x = -17203.45e3;     // m/s (from THEMIS SST, converted from km/s)
const double inflow_electron_velocity_y = 1169.93e3;      // m/s (from THEMIS SST, converted from km/s) 

// Ghost cell size
const int GHOST_CELLS = 2;

// Define the smoothDivergence function here
double smoothDivergence(double velocity_div) {
    double smoothing_factor = 0.9;  // Tune this value as needed
    return smoothing_factor * velocity_div;
}

// MPI domain decomposition parameters
int rank, size;                   // MPI rank and size
int start_x, end_x;               // Subdomain indices for each process

// Fields and current densities (local to each process)
std::vector<std::vector<double>> Ex, Ey, Bz, Bx, By, Jx, Jy, Jz;
std::vector<std::vector<double>> dPhi_dx, dPhi_dy;
std::vector<std::vector<double>> electron_velocity_x, electron_velocity_y, electron_pressure;
std::vector<double> electric_energy_history, magnetic_energy_history, kinetic_energy_history, thermal_energy_history, total_energy_history;
std::vector<std::vector<double>> ion_density, ion_velocity_x, ion_velocity_y, ion_pressure;
std::vector<std::vector<double>> particle_positions, particle_velocities;
std::vector<std::vector<double>> divergence(local_NX, std::vector<double>(NY));
std::vector<std::vector<double>> smoothed_divergence(local_NX, std::vector<double>(NY));
std::vector<std::vector<double>> field_x(local_NX, std::vector<double>(NY));
std::vector<std::vector<double>> field_y(local_NX, std::vector<double>(NY));
std::vector<std::vector<double>> phi(local_NX, std::vector<double>(NY));

// Fields for staggered grids
std::vector<std::vector<double>> Ex_staggered; // E_x at (i+1/2, j)
std::vector<std::vector<double>> Ey_staggered; // E_y at (i, j+1/2)
std::vector<std::vector<double>> Bx_staggered; // B_x at (i+1/2, j)
std::vector<std::vector<double>> By_staggered; // B_y at (i, j+1/2)
std::vector<std::vector<double>> Bz_centered;  // B_z at (i, j) - non-staggered (cell-centered)

// Current densities for staggered grids
std::vector<std::vector<double>> Jx_staggered; // J_x at (i+1/2, j)
std::vector<std::vector<double>> Jy_staggered; // J_y at (i, j+1/2)
std::vector<std::vector<double>> Jz_centered;  // J_z at (i, j) - non-staggered (cell-centered)

// Plasma and velocity properties (cell-centered, non-staggered grid)
std::vector<std::vector<double>> ion_density_centered;  // Ion density at (i, j)
std::vector<std::vector<double>> electron_pressure_centered; // Electron pressure at (i, j)
std::vector<std::vector<double>> electron_velocity_x_centered;
std::vector<std::vector<double>> electron_velocity_y_centered;
std::vector<std::vector<double>> ion_velocity_x_centered; // Ion velocity x at (i, j)
std::vector<std::vector<double>> ion_velocity_y_centered; // Ion velocity y at (i, j)
std::vector<std::vector<double>> ion_pressure_centered;

// Divergence fields
std::vector<std::vector<double>> divergence_centered(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));         // Divergence at (i, j)
std::vector<std::vector<double>> smoothed_divergence_centered(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0)); // Smoothed divergence at (i, j)

// Field potentials for divergence cleaning
std::vector<std::vector<double>> phi_centered; // Electric potential for cleaning divergence

// Function to print data every 100th timestep
void printDataMPI(const std::vector<std::vector<double>>& field, int timestep) {
    if (timestep % 100 == 0) { // Print every 100th timestep
        // Flatten the local field
        std::vector<double> local_flat_field;
        for (const auto& row : field) {
            local_flat_field.insert(local_flat_field.end(), row.begin(), row.end());
        }

        // Compute local sizes
        int local_size = local_flat_field.size();
        std::vector<int> recv_counts(size, 0); // Number of elements from each process
        std::vector<int> displacements(size, 0); // Start index for each process in the global array

        // Gather local sizes at rank 0
        MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Compute displacements
            for (int i = 1; i < size; ++i) {
                displacements[i] = displacements[i - 1] + recv_counts[i - 1];
            }
        }

        // Prepare the global array at rank 0
        std::vector<double> global_flat_field;
        if (rank == 0) {
            global_flat_field.resize(std::accumulate(recv_counts.begin(), recv_counts.end(), 0));
        }

        // Use MPI_Gatherv to gather data into global array
        MPI_Gatherv(local_flat_field.data(), local_size, MPI_DOUBLE,
                    global_flat_field.data(), recv_counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print the global field data
            std::cout << "Timestep: " << timestep << ", Field Data:" << std::endl;

            // Reconstruct and print the 2D field from the global flat field
            int global_index = 0;
            for (int i = 0; i < NX; ++i) {
                for (int j = 0; j < NY; ++j) {
                    std::cout << global_flat_field[global_index++] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

// Initialize MPI-related data structures and local subdomain sizes
void initializeMPI() {
    // Get the rank and size of the MPI world
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validate that px * py matches the total number of processes
    if (px * py != size) {
        if (rank == 0) {
            std::cerr << "Error: px * py (" << px * py << ") must equal the total number of MPI processes (" << size << ")." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Validate that NX and NY are evenly divisible by px and py
    if (NX % px != 0 || NY % py != 0) {
        if (rank == 0) {
            std::cerr << "Error: NX (" << NX << ") must be divisible by px (" << px << "), "
                      << "and NY (" << NY << ") must be divisible by py (" << py << ")." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate local domain size excluding ghost cells
    int subdomain_NX = NX / px;
    int subdomain_NY = NY / py;

    // Validate ghost cell dimensions
    if (2 * GHOST_CELLS >= subdomain_NX || 2 * GHOST_CELLS >= subdomain_NY) {
        if (rank == 0) {
            std::cerr << "Error: GHOST_CELLS (" << GHOST_CELLS << ") is too large for the subdomain size. "
                      << "subdomain_NX = " << subdomain_NX << ", subdomain_NY = " << subdomain_NY << "." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Include ghost cells for local computations
    local_NX = subdomain_NX + 2 * GHOST_CELLS;
    local_NY = subdomain_NY + 2 * GHOST_CELLS;

    // Synchronize critical parameters across ranks
    MPI_Bcast(&local_NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&GHOST_CELLS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Debug output for rank-specific domain sizes
    std::cout << "Rank " << rank << " Initialization Details:\n"
              << "  Global NX = " << NX << ", NY = " << NY << "\n"
              << "  Process grid px = " << px << ", py = " << py << "\n"
              << "  Subdomain subdomain_NX = " << subdomain_NX << ", subdomain_NY = " << subdomain_NY << "\n"
              << "  Local domain local_NX = " << local_NX << ", local_NY = " << local_NY << "\n"
              << "  GHOST_CELLS = " << GHOST_CELLS << std::endl;

    // Allocate local arrays with ghost cells
    try {
        Ex_staggered.assign(local_NX + 1, std::vector<double>(local_NY, 0.0));
        Ey_staggered.assign(local_NX, std::vector<double>(local_NY + 1, 0.0));
        Bx_staggered.assign(local_NX + 1, std::vector<double>(local_NY, 0.0));
        By_staggered.assign(local_NX, std::vector<double>(local_NY + 1, 0.0));
        Bz_centered.assign(local_NX, std::vector<double>(local_NY, 0.0));
        Jx_staggered.assign(local_NX + 1, std::vector<double>(local_NY, 0.0));
        Jy_staggered.assign(local_NX, std::vector<double>(local_NY + 1, 0.0));
        Jz_centered.assign(local_NX, std::vector<double>(local_NY, 0.0));
        ion_density_centered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        electron_pressure_centered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        ion_velocity_x_centered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        ion_velocity_y_centered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error on rank " << rank << ": Memory allocation failed. " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Ensure ghost cells are adequately handled
    if (local_NX < 2 * GHOST_CELLS || local_NY < 2 * GHOST_CELLS) {
        std::cerr << "Error on rank " << rank << ": Insufficient local domain size for ghost cells." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Final debug confirmation
    std::cout << "Rank " << rank << ": MPI initialization successful. Arrays allocated with ghost cells." << std::endl;
}

// Exchange ghost cells with neighboring processes for staggered grids
void exchangeGhostCellsMPI(
    std::vector<std::vector<double>>& field,
    const std::string& field_type,
    int local_NX, int local_NY,
    int rank, int size
) {
    assert(!field.empty() && field[0].size() > 0);

    int rows = field.size();
    int cols = field[0].size();

    // Ensure field dimensions are sufficient for ghost cell exchange
    if (rows < 2 * GHOST_CELLS || cols < 2 * GHOST_CELLS) {
        if (rank == 0) {
            std::cerr << "Error: Field dimensions insufficient for ghost cell exchange in "
                      << field_type << "." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Request requests[8]; // 4 for sends, 4 for receives
    int request_count = 0;
    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Exchange ghost cells in the x-direction
    if (prev_rank != MPI_PROC_NULL) {
        MPI_Isend(&field[GHOST_CELLS][0], cols, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[request_count++]); // Send to left
        MPI_Irecv(&field[0][0], cols, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[request_count++]); // Receive from left
    }
    if (next_rank != MPI_PROC_NULL) {
        MPI_Isend(&field[rows - 2 * GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[request_count++]); // Send to right
        MPI_Irecv(&field[rows - GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[request_count++]); // Receive from right
    }

    // Exchange ghost cells in the y-direction
    // Create a new MPI data type to represent a column of the field
    MPI_Datatype column_type;
    MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    int prev_rank_y = (rank < px) ? MPI_PROC_NULL : rank - px; // Rank above
    int next_rank_y = (rank >= size - px) ? MPI_PROC_NULL : rank + px; // Rank below

    if (prev_rank_y != MPI_PROC_NULL) {
        MPI_Isend(&field[0][GHOST_CELLS], 1, column_type, prev_rank_y, 1, MPI_COMM_WORLD, &requests[request_count++]); // Send to top
        MPI_Irecv(&field[0][0], 1, column_type, prev_rank_y, 1, MPI_COMM_WORLD, &requests[request_count++]); // Receive from top
    }
    if (next_rank_y != MPI_PROC_NULL) {
        MPI_Isend(&field[0][cols - 2 * GHOST_CELLS], 1, column_type, next_rank_y, 1, MPI_COMM_WORLD, &requests[request_count++]); // Send to bottom
        MPI_Irecv(&field[0][cols - GHOST_CELLS], 1, column_type, next_rank_y, 1, MPI_COMM_WORLD, &requests[request_count++]); // Receive from bottom
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    // Free the column type
    MPI_Type_free(&column_type);

    if (rank == 0) {
        std::cout << "Ghost cell exchange complete for field type: " << field_type << "." << std::endl;
    }
}

// Optimized Batch function to exchange ghost cells for multiple fields
void exchangeGhostCellsBatchMPI(
    std::vector<std::vector<std::vector<double>>>& fields,
    const std::vector<std::string>& field_types,
    int local_NX, int local_NY,
    int rank, int size
) {
    assert(fields.size() == field_types.size() && "Mismatch between fields and field types");

    size_t num_fields = fields.size();
    MPI_Request* requests = new MPI_Request[8 * num_fields]; // 4 sends and 4 receives for each field
    int request_count = 0;

    for (size_t i = 0; i < num_fields; ++i) {
        auto& field = fields[i];
        const std::string& field_type = field_types[i];

        int rows = field.size();
        int cols = field[0].size();

        // Ensure field dimensions are sufficient for ghost cell exchange
        if (rows < 2 * GHOST_CELLS || cols < 2 * GHOST_CELLS) {
            if (rank == 0) {
                std::cerr << "Error: Field dimensions insufficient for ghost cell exchange in "
                          << field_type << "." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int next_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // Exchange ghost cells in the x-direction
        if (prev_rank != MPI_PROC_NULL) {
            MPI_Isend(&field[GHOST_CELLS][0], cols, MPI_DOUBLE, prev_rank, i * 4, MPI_COMM_WORLD, &requests[request_count++]); // Send to left
            MPI_Irecv(&field[0][0], cols, MPI_DOUBLE, prev_rank, i * 4 + 1, MPI_COMM_WORLD, &requests[request_count++]); // Receive from left
        }
        if (next_rank != MPI_PROC_NULL) {
            MPI_Isend(&field[rows - 2 * GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, i * 4 + 1, MPI_COMM_WORLD, &requests[request_count++]); // Send to right
            MPI_Irecv(&field[rows - GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, i * 4, MPI_COMM_WORLD, &requests[request_count++]); // Receive from right
        }

        // Exchange ghost cells in the y-direction
        // Create a new MPI data type to represent a column of the field
        MPI_Datatype column_type;
        MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);

        int prev_rank_y = (rank < px) ? MPI_PROC_NULL : rank - px; // Rank above
        int next_rank_y = (rank >= size - px) ? MPI_PROC_NULL : rank + px; // Rank below

        if (prev_rank_y != MPI_PROC_NULL) {
            MPI_Isend(&field[0][GHOST_CELLS], 1, column_type, prev_rank_y, i * 4 + 2, MPI_COMM_WORLD, &requests[request_count++]); // Send to top
            MPI_Irecv(&field[0][0], 1, column_type, prev_rank_y, i * 4 + 3, MPI_COMM_WORLD, &requests[request_count++]); // Receive from top
        }
        if (next_rank_y != MPI_PROC_NULL) {
            MPI_Isend(&field[0][cols - 2 * GHOST_CELLS], 1, column_type, next_rank_y, i * 4 + 3, MPI_COMM_WORLD, &requests[request_count++]); // Send to bottom
            MPI_Irecv(&field[0][cols - GHOST_CELLS], 1, column_type, next_rank_y, i * 4 + 2, MPI_COMM_WORLD, &requests[request_count++]); // Receive from bottom
        }

        // Free the column type
        MPI_Type_free(&column_type);
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    delete[] requests;

    if (rank == 0) {
        std::cout << "Batch ghost cell exchange completed for all fields." << std::endl;
    }
}

void exchangeGhostCellsMPIAsync(
    std::vector<std::vector<double>>& field,
    const std::string& field_type,
    int local_NX, int local_NY,
    int rank, int size,
    MPI_Request* requests
) {
    assert(!field.empty() && field[0].size() > 0);

    int rows = field.size();
    int cols = field[0].size();

    // Ensure field dimensions are sufficient for ghost cell exchange
    if (rows < 2 * GHOST_CELLS || cols < 2 * GHOST_CELLS) {
        if (rank == 0) {
            std::cerr << "Error: Field dimensions insufficient for ghost cell exchange in field type: " 
                      << field_type << "." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Non-blocking communication for x-direction (left and right neighbors)
    MPI_Isend(&field[GHOST_CELLS][0], cols, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[0]); // Send to left
    MPI_Irecv(&field[0][0], cols, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &requests[1]); // Receive from left
    MPI_Isend(&field[rows - 2 * GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &requests[2]); // Send to right
    MPI_Irecv(&field[rows - GHOST_CELLS][0], cols, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[3]); // Receive from right

    // Create a new MPI data type to represent a column of the field
    MPI_Datatype column_type;
    MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Non-blocking communication for y-direction (top and bottom neighbors)
    int prev_rank_y = (rank < px) ? MPI_PROC_NULL : rank - px; // Rank above
    int next_rank_y = (rank >= size - px) ? MPI_PROC_NULL : rank + px; // Rank below

    if (prev_rank_y != MPI_PROC_NULL) {
        MPI_Isend(&field[0][GHOST_CELLS], 1, column_type, prev_rank_y, 2, MPI_COMM_WORLD, &requests[4]); // Send to top
        MPI_Irecv(&field[0][0], 1, column_type, prev_rank_y, 3, MPI_COMM_WORLD, &requests[5]); // Receive from top
    }
    if (next_rank_y != MPI_PROC_NULL) {
        MPI_Isend(&field[0][cols - 2 * GHOST_CELLS], 1, column_type, next_rank_y, 3, MPI_COMM_WORLD, &requests[6]); // Send to bottom
        MPI_Irecv(&field[0][cols - GHOST_CELLS], 1, column_type, next_rank_y, 2, MPI_COMM_WORLD, &requests[7]); // Receive from bottom
    }

    // Free the column type
    MPI_Type_free(&column_type);
}

// Function to wait and check for errors in non-blocking communication
void waitForCommunication(MPI_Request* requests) {
    MPI_Status statuses[4];

    // Wait for all non-blocking communication to complete
    MPI_Waitall(4, requests, statuses);

    // Check for errors
    for (int i = 0; i < 4; ++i) {
        if (statuses[i].MPI_ERROR != MPI_SUCCESS) {
            std::cerr << "Error: MPI communication failed in exchangeGhostCellsMPIAsync." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, statuses[i].MPI_ERROR);
        }
    }
}

// Compute divergence of a vector field
std::vector<std::vector<double>> computeDivergenceMPI(const std::vector<std::vector<double>>& field_x,
                                                      const std::vector<std::vector<double>>& field_y,
                                                      bool return_global = false) {
    // Local divergence computation
    std::vector<std::vector<double>> local_divergence(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

    // Compute divergence for local domain, excluding ghost cells
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            double dEx_dx = (field_x[i + 1][j] - field_x[i - 1][j]) / (2.0 * DX);
            double dEy_dy = (field_y[i][j + 1] - field_y[i][j - 1]) / (2.0 * DY);

            local_divergence[i][j] = dEx_dx + dEy_dy;
        }
    }

    // Synchronize ghost cells for divergence
    exchangeGhostCellsMPI(local_divergence, "divergence", local_NX, local_NY, rank, size);

    // Optionally compute the global divergence on rank 0
    if (return_global) {
        std::vector<double> flat_local(local_NX * local_NY, 0.0);
        int idx = 0;
        for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
            for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
                flat_local[idx++] = local_divergence[i][j];
            }
        }

        std::vector<double> flat_global(NX * NY, 0.0);
        MPI_Reduce(flat_local.data(), flat_global.data(), local_NX * local_NY, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Reconstruct the global divergence on rank 0
        if (rank == 0) {
            std::vector<std::vector<double>> global_divergence(NX, std::vector<double>(NY, 0.0));
            idx = 0;
            for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
                for (int j = GHOST_CELLS; j < NY + GHOST_CELLS; ++j) {
                    global_divergence[i][j] = flat_global[idx++];
                }
            }
            return global_divergence;
        }
    }

    // Return local divergence
    return local_divergence;
}

// Restriction: Coarse-graining the grid
std::vector<std::vector<double>> restrictGridMPI(const std::vector<std::vector<double>>& fine_grid,
                                                 int local_NX, int local_NY,
                                                 const std::string& field_type,
                                                 int rank, int size) {
    // Calculate coarse grid size (local)
    int local_coarse_NX = (local_NX - 2 * GHOST_CELLS) / 2 + 2 * GHOST_CELLS;
    int local_coarse_NY = (local_NY - 2 * GHOST_CELLS) / 2 + 2 * GHOST_CELLS;
    std::vector<std::vector<double>> coarse_grid(local_coarse_NX, std::vector<double>(local_coarse_NY, 0.0));

    // Perform restriction on local domain
    for (int i = GHOST_CELLS; i < local_coarse_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_coarse_NY - GHOST_CELLS; ++j) {
            // Map coarse grid indices to fine grid indices
            int fi = 2 * (i - GHOST_CELLS) + GHOST_CELLS;
            int fj = 2 * (j - GHOST_CELLS) + GHOST_CELLS;

            // Calculate contributions from neighbors on the fine grid
            double center = fine_grid[fi][fj];
            double x_neighbors = 0.0, y_neighbors = 0.0, diagonal_neighbors = 0.0;

            // Use local_NX and local_NY to ensure you stay within the bounds of the local fine grid
            if (fi + 1 < local_NX - GHOST_CELLS) x_neighbors += fine_grid[fi + 1][fj];
            if (fi - 1 >= GHOST_CELLS) x_neighbors += fine_grid[fi - 1][fj];
            if (fj + 1 < local_NY + 2 * GHOST_CELLS) y_neighbors += fine_grid[fi][fj + 1];
            if (fj - 1 >= GHOST_CELLS) y_neighbors += fine_grid[fi][fj - 1];

            if (fi + 1 < local_NX - GHOST_CELLS && fj + 1 < local_NY + 2 * GHOST_CELLS) diagonal_neighbors += fine_grid[fi + 1][fj + 1];
            if (fi + 1 < local_NX - GHOST_CELLS && fj - 1 >= GHOST_CELLS) diagonal_neighbors += fine_grid[fi + 1][fj - 1];
            if (fi - 1 >= GHOST_CELLS && fj + 1 < local_NY + 2 * GHOST_CELLS) diagonal_neighbors += fine_grid[fi - 1][fj + 1];
            if (fi - 1 >= GHOST_CELLS && fj - 1 >= GHOST_CELLS) diagonal_neighbors += fine_grid[fi - 1][fj - 1];

            // Restrict to coarse grid
            coarse_grid[i][j] = 0.25 * center +
                                0.125 * (x_neighbors + y_neighbors) +
                                0.0625 * diagonal_neighbors;
        }
    }

    // Communicate ghost cells for the coarse grid
    exchangeGhostCellsMPI(coarse_grid, field_type, local_NX / 2, local_NY / 2, rank, size);

    // Debugging: Confirm restriction
    if (rank == 0) {
        std::cout << "Restriction completed on rank " << rank << " for field type: " << field_type << "." << std::endl;
    }

    return coarse_grid;
}

// Prolongation
void prolongateGridMPI(const std::vector<std::vector<double>>& coarse_grid, 
                       std::vector<std::vector<double>>& fine_grid, 
                       int local_NX, int local_NY, 
                       const std::string& field_type, 
                       int rank, int size) {
    int local_coarse_NX = (local_NX - 2 * GHOST_CELLS) / 2 + 2 * GHOST_CELLS; // Local coarse NX with ghost cells
    int local_coarse_NY = (local_NY - 2 * GHOST_CELLS) / 2 + 2 * GHOST_CELLS; // Coarse NY with ghost cells

    // Perform bilinear interpolation for local domain
    for (int i = GHOST_CELLS; i < local_coarse_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_coarse_NY - GHOST_CELLS; ++j) {
            int fi = 2 * (i - GHOST_CELLS) + GHOST_CELLS;
            int fj = 2 * (j - GHOST_CELLS) + GHOST_CELLS;

            // Assign values to fine grid using bilinear interpolation
            fine_grid[fi][fj] = coarse_grid[i][j];
            if (fi + 1 < local_NX - GHOST_CELLS) {
                fine_grid[fi + 1][fj] = 0.5 * (coarse_grid[i][j] + coarse_grid[i + 1][j]);
            }
            if (fj + 1 < local_NY + GHOST_CELLS) {
                fine_grid[fi][fj + 1] = 0.5 * (coarse_grid[i][j] + coarse_grid[i][j + 1]);
            }
            if (fi + 1 < local_NX - GHOST_CELLS && fj + 1 < local_NY + GHOST_CELLS) {
                fine_grid[fi + 1][fj + 1] = 0.25 * (coarse_grid[i][j] + coarse_grid[i + 1][j] +
                                                    coarse_grid[i][j + 1] + coarse_grid[i + 1][j + 1]);
            }
        }
    }

    // Communicate ghost cells in the fine grid
    exchangeGhostCellsMPI(fine_grid, field_type, local_NX, local_NY, rank, size);

    // Debugging: Confirm prolongation
    if (rank == 0) {
        std::cout << "Prolongation completed on rank " << rank << " for field type: " << field_type << "." << std::endl;
    }
}

// Parallel Gauss-Seidel Relaxation
void gaussSeidelRelaxationMPI(std::vector<std::vector<double>>& grid,
                              const std::vector<std::vector<double>>& rhs,
                              int local_NX, int local_NY,
                              const std::string& field_type,
                              int rank, int size,
                              int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Communicate ghost cells
        exchangeGhostCellsMPI(grid, field_type, local_NX, local_NY, rank, size);

        // Perform Gauss-Seidel iteration within the local domain (excluding ghost cells)
        for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
            for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
                grid[i][j] = 0.25 * (rhs[i][j] +
                                     grid[i + 1][j] + grid[i - 1][j] +
                                     grid[i][j + 1] + grid[i][j - 1]);
            }
        }
    }
}

std::vector<std::vector<double>> solvePoissonMultigridMPI(const std::vector<std::vector<double>>& rhs,
                                                          int local_NX, int local_NY, int levels,
                                                          const std::string& field_type,
                                                          int rank, int size) {
    // Validate grid dimensions
    if ((local_NX - 2 * GHOST_CELLS) % (1 << levels) != 0 || (local_NY - 2 * GHOST_CELLS) % (1 << levels) != 0) {
        if (rank == 0) {
            std::cerr << "Error: Grid dimensions are not compatible with the multigrid levels. "
                      << "Ensure NX and NY are powers of 2 or adjust the number of levels." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<std::vector<double>> solution(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

    // Base case: Coarsest level solve
    if (levels == 1 || (local_NX - 2 * GHOST_CELLS == 2 && local_NY - 2 * GHOST_CELLS == 2)) {
        gaussSeidelRelaxationMPI(solution, rhs, local_NX, local_NY, field_type, rank, size, 10);
        return solution;
    }

    // Step 1: Relaxation
    gaussSeidelRelaxationMPI(solution, rhs, local_NX, local_NY, field_type, rank, size, 5);

    // Synchronize ghost cells
    exchangeGhostCellsMPI(solution, field_type, local_NX, local_NY, rank, size);

    // Step 2: Compute residual
    std::vector<std::vector<double>> residual(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            residual[i][j] = rhs[i][j] -
                             (4 * solution[i][j] -
                              solution[i + 1][j] - solution[i - 1][j] -
                              solution[i][j + 1] - solution[i][j - 1]);
        }
    }

    // Step 3: Restrict residual to coarser grid
    auto coarse_residual = restrictGridMPI(residual, local_NX, local_NY, field_type, rank, size);

    // Step 4: Solve coarse problem recursively
    auto coarse_solution = solvePoissonMultigridMPI(coarse_residual, local_NX / 2, local_NY / 2, levels - 1, field_type, rank, size);

    // Step 5: Prolongate coarse solution
    prolongateGridMPI(coarse_solution, solution, local_NX, local_NY, field_type, rank, size);

    // Synchronize ghost cells again
    exchangeGhostCellsMPI(solution, field_type, local_NX, local_NY, rank, size);

    // Step 6: Final relaxation
    gaussSeidelRelaxationMPI(solution, rhs, local_NX, local_NY, field_type, rank, size, 5);

    return solution;
}

// Step 1: Define the Domain
void initializeDomain() {
    std::cout << "Initializing domain with NX = " << NX << ", NY = " << NY << std::endl;
    // The domain setup is already defined with NX, NY, DX, DY, and other global constants.
}

// Step 2: Set Up Initial Conditions
void initializeMagneticField(
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    std::vector<std::vector<double>>& Bz_centered) {

    double dipole_moment = 8.0e15;  // Dipole moment (T*m³) for Earth's magnetic field
    double dipole_x = NX * DX / 2.0; // Dipole center x-coordinate
    double dipole_y = NY * DY / 2.0; // Dipole center y-coordinate

    // Calculate starting indices for each rank's subdomain
    int start_x_index = (rank % px) * (NX / px);
    int start_y_index = (rank / px) * (NY / py);

    // Initialize Dipole Magnetic Field on the staggered grid
    for (int i = 0; i < Bx_staggered.size(); ++i) {
        for (int j = 0; j < Bx_staggered[0].size(); ++j) {
            // Compute physical coordinates for Bx_staggered at x-faces
            double x_Bx = start_x_index * DX + (i - GHOST_CELLS + 0.5) * DX;
            double y_Bx = start_y_index * DY + (j - GHOST_CELLS) * DY;

            // Calculate distance to dipole center
            double r_Bx = std::sqrt((x_Bx - dipole_x) * (x_Bx - dipole_x) + (y_Bx - dipole_y) * (y_Bx - dipole_y));

            // Dipole field equations for Bx_staggered
            double Bx_dipole = -dipole_moment * (3 * (x_Bx - dipole_x) * (y_Bx - dipole_y)) / (std::pow(r_Bx, 5));

            // Assign to Bx_staggered
            Bx_staggered[i][j] = Bx_dipole;
        }
    }

    for (int i = 0; i < By_staggered.size(); ++i) {
        for (int j = 0; j < By_staggered[0].size(); ++j) {
            // Compute physical coordinates for By_staggered at y-faces
            double x_By = start_x_index * DX + (i - GHOST_CELLS) * DX;
            double y_By = start_y_index * DY + (j - GHOST_CELLS + 0.5) * DY;

            // Calculate distance to dipole center
            double r_By = std::sqrt((x_By - dipole_x) * (x_By - dipole_x) + (y_By - dipole_y) * (y_By - dipole_y));

            // Dipole field equations for By_staggered
            double By_dipole = dipole_moment * (2 * std::pow(y_By - dipole_y, 2) - std::pow(x_By - dipole_x, 2)) / (std::pow(r_By, 5));

            // Assign to By_staggered
            By_staggered[i][j] = By_dipole;
        }
    }

    // Bz_centered is initialized to zero as it is a 2D simulation in the x-y plane
    for (int i = 0; i < Bz_centered.size(); ++i) {
        for (int j = 0; j < Bz_centered[0].size(); ++j) {
            Bz_centered[i][j] = 0.0;
        }
    }

    // Apply Solar Wind Magnetic Field to Bx_staggered and By_staggered
    for (int i = 0; i < Bx_staggered.size(); ++i) {
        for (int j = 0; j < Bx_staggered[0].size(); ++j) {
            double x_Bx = start_x_index * DX + (i - GHOST_CELLS + 0.5) * DX;
            double y_Bx = start_y_index * DY + (j - GHOST_CELLS) * DY;

            if (x_Bx < NX * DX / 4) { // Solar wind occupies the left quarter of the domain
                Bx_staggered[i][j] += inflow_magnetic_field_x;
            }
        }
    }

    for (int i = 0; i < By_staggered.size(); ++i) {
        for (int j = 0; j < By_staggered[0].size(); ++j) {
            double x_By = start_x_index * DX + (i - GHOST_CELLS) * DX;
            double y_By = start_y_index * DY + (j - GHOST_CELLS + 0.5) * DY;

            if (x_By < NX * DX / 4) { // Solar wind occupies the left quarter of the domain
                By_staggered[i][j] += inflow_magnetic_field_y;
            }
        }
    }

    std::cout << "Magnetic field initialized successfully on rank " << rank << "." << std::endl;
}

void initializePlasmaProperties(
    std::vector<std::vector<double>>& ion_density_centered,
    std::vector<std::vector<double>>& ion_velocity_x_centered,
    std::vector<std::vector<double>>& ion_velocity_y_centered,
    std::vector<std::vector<double>>& electron_pressure_centered) {

    // Starting indices for each rank's subdomain
    int start_x_index = (rank % px) * (NX / px);

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Convert local index to global index for checking the region
            int global_i = start_x_index + (i - GHOST_CELLS);

            if (global_i < NX / 4) { // Solar wind region
                ion_density_centered[i][j] = inflow_electron_density;
                ion_velocity_x_centered[i][j] = inflow_electron_velocity_x;
                ion_velocity_y_centered[i][j] = inflow_electron_velocity_y;
                electron_pressure_centered[i][j] = k_B * inflow_electron_density * inflow_electron_temperature;
            } else { // Magnetosphere region
                ion_density_centered[i][j] = density_1;
                ion_velocity_x_centered[i][j] = 0.0;
                ion_velocity_y_centered[i][j] = 0.0;
                electron_pressure_centered[i][j] = density_1 * k_B * 264.31; // Example ion temperature in K
            }
        }
    }

    std::cout << "Plasma properties initialized successfully on rank " << rank << std::endl;
}

void initializeCurrentSheet(
    std::vector<std::vector<double>>& Jx_staggered,
    std::vector<std::vector<double>>& Jy_staggered,
    int local_NX, int local_NY, int rank, double CURRENT_SHEET_Y, double CURRENT_SHEET_WIDTH) {

    // Calculate starting y index for the current rank
    int start_y_index = (rank / px) * (NY / py);

    // Iterate over the local grid, considering staggered positions
    for (int i = 0; i < local_NX + 1; ++i) { // Jx is at x-faces
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) { // Jx has ghost cells in y-direction
            // Calculate physical y-coordinate for Jx
            double y = start_y_index * DY + (j - GHOST_CELLS) * DY;

            // Define the current sheet region for Jx
            if (std::abs(y - CURRENT_SHEET_Y) < CURRENT_SHEET_WIDTH / 2.0) {
                if (i < Jx_staggered.size() && j < Jx_staggered[i].size()) {
                    Jx_staggered[i][j] = 1.0e-6; // Example current density in A/m²
                }
            }
        }
    }

    for (int i = 0; i < local_NX; ++i) { // Jy is at y-faces
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS + 1; ++j) {
            // Calculate physical y-coordinate for Jy
            double y = start_y_index * DY + (j - GHOST_CELLS + 0.5) * DY;

            // Define the current sheet region for Jy
            if (std::abs(y - CURRENT_SHEET_Y) < CURRENT_SHEET_WIDTH / 2.0) {
                if (i < Jy_staggered.size() && j < Jy_staggered[i].size()) {
                    Jy_staggered[i][j] = -1.0e-6; // Oppositely directed current
                }
            }
        }
    }

    std::cout << "Current sheet initialized successfully on rank " << rank << "." << std::endl;
}

// Step 3: Apply Boundary Conditions
void applyStaggeredBoundaryConditionsMPI(std::vector<std::vector<double>>& Bx_staggered,
                             std::vector<std::vector<double>>& By_staggered,
                             std::vector<std::vector<double>>& Bz_centered,
                             std::vector<std::vector<double>>& Ex_staggered,
                             std::vector<std::vector<double>>& Ey_staggered,
                             const std::vector<std::vector<double>>& ion_density_centered,
                             const std::vector<std::vector<double>>& ion_velocity_x_centered,
                             const std::vector<std::vector<double>>& ion_velocity_y_centered,
                             int local_NX, int local_NY, int rank, int size) {
    applyStaggeredBoundaryConditionsMPI(Bx_staggered, By_staggered, Bz_centered, Ex_staggered, Ey_staggered, ion_density_centered, ion_velocity_x_centered, ion_velocity_y_centered, local_NX, local_NY, rank, size);
}

void initializeCTGridStructure(int local_NX, int local_NY);
void initializeStaggeredFields(std::vector<std::vector<double>>& Bx_staggered,
                              std::vector<std::vector<double>>& By_staggered,
                              std::vector<std::vector<double>>& Bz_centered,
                              int local_NX, int local_NY, int rank,
                              double CURRENT_SHEET_Y, double CURRENT_SHEET_THICKNESS,
                              double B0, double dipole_center_x, double dipole_center_y, double dipole_moment);
void initializeElectricFields(std::vector<std::vector<double>>& Ex_staggered,
                              std::vector<std::vector<double>>& Ey_staggered,
                              int local_NX, int local_NY);
void initializeFluids();
void initializeParticles();
void initializeElectronFields();

// Master Initialization Function
void initializeSimulation(int local_NX, int local_NY, int rank, int size,
                          double CURRENT_SHEET_Y, double CURRENT_SHEET_THICKNESS,
                          double B0, double dipole_center_x, double dipole_center_y, double dipole_moment) {
    // Initialize domain
    initializeDomain();

    // Initialize staggered grid structure
    initializeCTGridStructure(local_NX, local_NY);

    // Initialize magnetic fields (staggered)
    initializeStaggeredFields(Bx_staggered, By_staggered, Bz_centered,
                              local_NX, local_NY, rank,
                              CURRENT_SHEET_Y, CURRENT_SHEET_THICKNESS, B0,
                              dipole_center_x, dipole_center_y, dipole_moment);

    // Initialize electric fields (staggered)
    initializeElectricFields(Ex_staggered, Ey_staggered, local_NX, local_NY);

    // Initialize plasma properties (centered)
    initializeFluids();

    // Initialize electron fields (centered)
    initializeElectronFields();

    // Initialize particles
    initializeParticles();

    // Apply boundary conditions (use staggered variables)
    applyStaggeredBoundaryConditionsMPI(Bx_staggered, By_staggered, Bz_centered,
                                        Ex_staggered, Ey_staggered,
                                        ion_density_centered, ion_velocity_x_centered, ion_velocity_y_centered,
                                        local_NX, local_NY, rank, size);

    // Debug output to confirm initialization
    std::cout << "Rank " << rank << ": Fields and particles initialized and synchronized." << std::endl;
}

// Compute Staggered Curl
void computeStaggeredCurl(
    const std::vector<std::vector<double>>& Ex_staggered,    // E_x at (i+1/2, j)
    const std::vector<std::vector<double>>& Ey_staggered,    // E_y at (i, j+1/2)
    std::vector<std::vector<double>>& Bz_centered,          // B_z at (i, j)
    const std::vector<std::vector<double>>& ion_density_centered, // Ion density at (i, j)
    const std::vector<std::vector<double>>& Jx_staggered,    // Current density x-component at (i+1/2, j)
    const std::vector<std::vector<double>>& Jy_staggered,    // Current density y-component at (i, j+1/2)
    const std::vector<std::vector<double>>& Bx_staggered,    // Magnetic field x-component at (i+1/2, j)
    const std::vector<std::vector<double>>& By_staggered,    // Magnetic field y-component at (i, j+1/2)
    int local_NX, int local_NY, int rank, int size)
{
    // Update Bz_centered (defined at cell centers)
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Ensure indices are within bounds for staggered arrays
            if (i < 0 || i >= Bz_centered.size() || j < 0 || j >= Bz_centered[0].size()) {
                continue; // Skip if out of bounds
            }

            double dEy_dx, dEx_dy;

            // 1. Compute Curl of E
            // dEy_dx at (i, j) using Ey_staggered at (i, j+1/2) and (i-1, j+1/2)
            if (i > 0 && i < local_NX && j < Ey_staggered[0].size()) {
                dEy_dx = (Ey_staggered[i][j] - Ey_staggered[i - 1][j]) / DX;
            } else {
                dEy_dx = 0.0; // Boundary condition or extrapolation as needed
            }

            // dEx_dy at (i, j) using Ex_staggered at (i+1/2, j) and (i+1/2, j-1)
            if (j > 0 && j < local_NY + 2 * GHOST_CELLS && i < Ex_staggered.size()) {
                dEx_dy = (Ex_staggered[i][j] - Ex_staggered[i][j - 1]) / DY;
            } else {
                dEx_dy = 0.0; // Boundary condition or extrapolation as needed
            }

            // 2. Hall Term Contribution
            // Ensure density is not zero and interpolate J and B to cell centers
            double density = std::max(ion_density_centered[i][j], 1e-10);

            // Interpolate Jx to (i, j)
            double Jx_center = 0.0;
            if (i > 0 && i < Jx_staggered.size())
                Jx_center = 0.5 * (Jx_staggered[i][j] + Jx_staggered[i - 1][j]);

            // Interpolate Jy to (i, j)
            double Jy_center = 0.0;
            if (j > 0 && j < local_NY + 2 * GHOST_CELLS && i < Jy_staggered.size())
                Jy_center = 0.5 * (Jy_staggered[i][j] + Jy_staggered[i][j - 1]);

            // Interpolate Bx to (i, j)
            double Bx_center = 0.0;
            if (i > 0 && i < Bx_staggered.size())
                Bx_center = 0.5 * (Bx_staggered[i][j] + Bx_staggered[i - 1][j]);

            // Interpolate By to (i, j)
            double By_center = 0.0;
            if (j > 0 && j < By_staggered[0].size())
                By_center = 0.5 * (By_staggered[i][j] + By_staggered[i][j - 1]);

            // Bz is already at the cell center
            double Bz_val = Bz_centered[i][j];

            // Compute JxB terms at cell center
            double J_cross_B_x = Jy_center * Bz_val - Jx_center * By_center;
            double J_cross_B_y = Jx_center * Bz_val - Jy_center * Bx_center;

            // Compute divergence of JxB (Hall term) using centered differences
            double Hall_dJxB_dx = 0.0;
            if (i > 0 && i < local_NX)
                Hall_dJxB_dx = (J_cross_B_x - J_cross_B_x) / DX;

            double Hall_dJxB_dy = 0.0;
            if (j > 0 && j < local_NY + 2 * GHOST_CELLS)
                Hall_dJxB_dy = (J_cross_B_y - J_cross_B_y) / DY;

            double hall_term = (Hall_dJxB_dx + Hall_dJxB_dy) / (QE * density);

            // 3. Update Bz using Faraday's Law + Hall term
            Bz_centered[i][j] -= DT * (dEy_dx - dEx_dy + hall_term);
        }
    }

    // Exchange ghost cells for Bz_centered after updating
    exchangeGhostCellsMPI(Bz_centered, "Bz_centered", local_NX, local_NY, rank, size);
}

// Initialize Staggered Fields
void initializeStaggeredFields(
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    std::vector<std::vector<double>>& Bz_centered,
    int local_NX, int local_NY, int rank,
    double CURRENT_SHEET_Y, double CURRENT_SHEET_THICKNESS,
    double B0, double dipole_center_x, double dipole_center_y, double dipole_moment
) {
    // Error checking at the start
    if (Bx_staggered.empty() || By_staggered.empty() || Bz_centered.empty()) {
        std::cerr << "Error: Magnetic field components are empty on rank " << rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (Bx_staggered.size() != local_NX + 1 || Bx_staggered[0].size() != local_NY + 2 * GHOST_CELLS ||
        By_staggered.size() != local_NX || By_staggered[0].size() != local_NY + 2 * GHOST_CELLS + 1 ||
        Bz_centered.size() != local_NX || Bz_centered[0].size() != local_NY + 2 * GHOST_CELLS) {
        std::cerr << "Error: Magnetic field component sizes are incorrect on rank " << rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate starting x and y indices based on rank
    int start_x_index = (rank % px) * (NX / px);
    int start_y_index = (rank / px) * (NY / py);

    // Validate start indices
    if (start_x_index < 0 || start_y_index < 0 || start_x_index >= NX || start_y_index >= NY) {
        std::cerr << "Error: Starting indices out of bounds on rank " << rank
                  << " (start_x_index: " << start_x_index << ", start_y_index: " << start_y_index << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Iterate over the local grid for Bx_staggered, including ghost cells
    for (int i = 0; i < Bx_staggered.size(); ++i) {
        for (int j = 0; j < Bx_staggered[0].size(); ++j) {
            // Compute physical coordinates for Bx_staggered at x-faces
            double x_Bx = (start_x_index + (i - GHOST_CELLS + 0.5)) * DX;
            double y_Bx = (start_y_index + (j - GHOST_CELLS)) * DY;

            // Skip if outside the valid domain
            if (x_Bx < 0 || x_Bx > NX * DX || y_Bx < 0 || y_Bx > NY * DY) continue;

            // Displacement and distance from dipole center for Bx_staggered
            double dx_Bx = x_Bx - dipole_center_x;
            double dy_Bx = y_Bx - dipole_center_y;
            double r_Bx = std::max(std::sqrt(dx_Bx * dx_Bx + dy_Bx * dy_Bx), 1e-6);

            // Dipole field components
            double Br_Bx = (3.0 * dipole_moment * dx_Bx * dy_Bx) / std::pow(r_Bx, 5);
            double Bt_Bx = (dipole_moment * (2 * dy_Bx * dy_Bx - dx_Bx * dx_Bx)) / std::pow(r_Bx, 5);
            double Bx_dipole = Br_Bx * (dx_Bx / r_Bx) - Bt_Bx * (dy_Bx / r_Bx);

            // Solar wind and current sheet contributions
            double Bx_solarwind = (y_Bx > CURRENT_SHEET_Y) ? B0 : -B0;
            double Bx_current_sheet = B0 * tanh((y_Bx - CURRENT_SHEET_Y) / CURRENT_SHEET_THICKNESS);

            // Combine contributions
            Bx_staggered[i][j] = Bx_dipole - Bx_solarwind + Bx_current_sheet;
        }
    }

    // Repeat similar logic for By_staggered and Bz_centered
    // By_staggered
    for (int i = 0; i < By_staggered.size(); ++i) {
        for (int j = 0; j < By_staggered[0].size(); ++j) {
            double x_By = (start_x_index + (i - GHOST_CELLS)) * DX;
            double y_By = (start_y_index + (j - GHOST_CELLS + 0.5)) * DY;

            if (x_By < 0 || x_By > NX * DX || y_By < 0 || y_By > NY * DY) continue;

            double dx_By = x_By - dipole_center_x;
            double dy_By = y_By - dipole_center_y;
            double r_By = std::max(std::sqrt(dx_By * dx_By + dy_By * dy_By), 1e-6);

            double Br_By = (2.0 * dipole_moment * dx_By) / std::pow(r_By, 4);
            double Bt_By = (dipole_moment * dy_By) / std::pow(r_By, 4);
            double By_dipole = Br_By * (dy_By / r_By) + Bt_By * (dx_By / r_By);

            By_staggered[i][j] = By_dipole;
        }
    }

    // Bz_centered
    for (int i = 0; i < Bz_centered.size(); ++i) {
        for (int j = 0; j < Bz_centered[0].size(); ++j) {
            double x_Bz = (start_x_index + (i - GHOST_CELLS)) * DX;
            double y_Bz = (start_y_index + (j - GHOST_CELLS)) * DY;

            if (x_Bz < 0 || x_Bz > NX * DX || y_Bz < 0 || y_Bz > NY * DY) continue;

            Bz_centered[i][j] = 0.0; // Initialized to zero
        }
    }

    std::cout << "Staggered fields initialized successfully on rank " << rank << "." << std::endl;
}

//Initialize functions
void initializeElectricFields(std::vector<std::vector<double>>& Ex_staggered,
                              std::vector<std::vector<double>>& Ey_staggered,
                              int local_NX, int local_NY) {

    // Initialize Ex_staggered (defined at x-faces)
    for (int i = 0; i < local_NX + 1; ++i) { 
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
            Ex_staggered[i][j] = Egsm_x_1; // Initialize with the given value
        }
    }

    // Initialize Ey_staggered (defined at y-faces)
    for (int i = 0; i < local_NX; ++i) {         
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS + 1; ++j) { 
            Ey_staggered[i][j] = Egsm_y_1; // Initialize with the given value
        }
    }

    // Debug output to confirm initialization
    std::cout << "Rank " << rank << ": Electric fields Ex_staggered and Ey_staggered initialized successfully on staggered grid." << std::endl;
}

void initializeFluids() {
    try {
        // Validate field sizes to ensure consistency
        if (ion_density_centered.size() != local_NX || 
            (local_NX > 0 && ion_density_centered[0].size() != local_NY + 2 * GHOST_CELLS)) {
            throw std::runtime_error("Ion density size mismatch. Expected: " +
                                     std::to_string(local_NY + 2 * GHOST_CELLS) +
                                     " columns, but got: " + 
                                     std::to_string(ion_density_centered.empty() ? 0 : ion_density_centered[0].size()));
        }
        if (ion_pressure_centered.size() != local_NX || 
            (local_NX > 0 && ion_pressure_centered[0].size() != local_NY + 2 * GHOST_CELLS)) {
            throw std::runtime_error("Ion pressure size mismatch. Expected: " +
                                     std::to_string(local_NY + 2 * GHOST_CELLS) +
                                     " columns, but got: " +
                                     std::to_string(ion_pressure_centered.empty() ? 0 : ion_pressure_centered[0].size()));
        }

        // Initialize fluid properties
        for (int i = 0; i < local_NX; ++i) {
            for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
                ion_density_centered[i][j] = inflow_electron_density;
                ion_pressure_centered[i][j] = inflow_electron_density * k_B * inflow_temperature;
            }
        }

        std::cout << "Rank " << rank << ": Fluid properties initialized successfully." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error on rank " << rank << ": Memory allocation failed during fluid initialization. "
                  << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initializeParticles() {
    double avg_density = density_1;

    // Calculate the number of particles to initialize locally based on the local domain size.
    int local_particles = static_cast<int>(avg_density * local_NX * local_NY * DX * DY);

    // Reserve memory to avoid frequent reallocations.
    particle_positions.reserve(local_particles);
    particle_velocities.reserve(local_particles);

    // Calculate start_y for the current rank
    int start_y_index = (rank / px);
    double start_y = start_y_index * (NY / py) * DY; 

    for (int p = 0; p < local_particles; ++p) {
        // Calculate x-coordinate uniformly across the local subdomain, excluding ghost cells.
        double x = (start_x + GHOST_CELLS) * DX + std::rand() / double(RAND_MAX) * (local_NX - 2 * GHOST_CELLS) * DX;

        // Calculate y-coordinate uniformly across the local subdomain, excluding ghost cells.
        double y_min = start_y + GHOST_CELLS * DY;
        double y_max = start_y + (local_NY - GHOST_CELLS) * DY;
        double y = y_min + (std::rand() / double(RAND_MAX)) * (y_max - y_min);

        // Calculate random velocities for the particles.
        double vx = std::sqrt(264.31) * (std::rand() / double(RAND_MAX) - 0.5);
        double vy = std::sqrt(264.31) * (std::rand() / double(RAND_MAX) - 0.5);

        // Add the newly created particle's position and velocity to the respective vectors.
        particle_positions.push_back({x, y});
        particle_velocities.push_back({vx, vy});
    }

    // Ensure that the number of particle positions matches the number of particle velocities.
    if (particle_positions.size() != particle_velocities.size()) {
        std::cerr << "Error on rank " << rank << ": Particle positions and velocities mismatch." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initializeElectronFields() {
    // Validate field sizes for electrons using _centered variables
    try {
        if (electron_velocity_x_centered.size() != local_NX || 
            (local_NX > 0 && electron_velocity_x_centered[0].size() != local_NY + 2 * GHOST_CELLS)) {
            throw std::runtime_error("Electron velocity_x_centered size mismatch with local domain.");
        }

        if (electron_velocity_y_centered.size() != local_NX || 
            (local_NX > 0 && electron_velocity_y_centered[0].size() != local_NY + 2 * GHOST_CELLS)) {
            throw std::runtime_error("Electron velocity_y_centered size mismatch with local domain.");
        }

        if (electron_pressure_centered.size() != local_NX || 
            (local_NX > 0 && electron_pressure_centered[0].size() != local_NY + 2 * GHOST_CELLS)) {
            throw std::runtime_error("Electron pressure_centered size mismatch with local domain.");
        }

        // Initialize fields, iterating over the entire local domain including ghost cells
        for (int i = 0; i < local_NX; ++i) {
            for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
                electron_velocity_x_centered[i][j] = 0.0; // Assuming initial zero velocity
                electron_velocity_y_centered[i][j] = 0.0; // Assuming initial zero velocity
                electron_pressure_centered[i][j] = 1e-5; // Small initial pressure
            }
        }

        std::cout << "Rank " << rank << ": Electron fields initialized successfully." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error on rank " << rank << ": Memory allocation failed during electron field initialization. "
                  << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initializeElectricFields(std::vector<std::vector<double>>& Ex_staggered, 
                              std::vector<std::vector<double>>& Ey_staggered, 
                              int local_NX, int local_NY);

void initializeFluids();

void initializeParticles();

void initializeElectronFields();

// Function for applying staggered boundary conditions
void applyStaggeredBoundaryConditionsMPI(
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    std::vector<std::vector<double>>& Bz_centered,
    std::vector<std::vector<double>>& Ex_staggered,
    std::vector<std::vector<double>>& Ey_staggered,
    std::vector<std::vector<double>>& ion_density_centered,
    std::vector<std::vector<double>>& ion_velocity_x_centered,
    std::vector<std::vector<double>>& ion_velocity_y_centered,
    int local_NX, int local_NY, int rank, int size)
{
    const int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int next_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Request requests[24]; // Increased to accommodate more requests
    MPI_Status statuses[24];

    // Inflow Boundary Conditions (Left Boundary - Rank 0)
    if (rank == 0) {
        // Reinitialize electric fields, fluids, particles, and electron fields for inflow
        initializeElectricFields(Ex_staggered, Ey_staggered, local_NX, local_NY);
        initializeFluids();
        initializeParticles();
        initializeElectronFields();

        for (int j = 0; j < local_NY; ++j) {
            for (int gc = 0; gc < GHOST_CELLS; ++gc) {
                int i = GHOST_CELLS - 1 - gc; // Index for the leftmost ghost cells

                // Set inflow boundary conditions for Bx_staggered
                if (i >= 0 && i < Bx_staggered.size() && j < Bx_staggered[i].size())
                    Bx_staggered[i][j] = inflow_magnetic_field_x;

                // Set inflow boundary conditions for By_staggered
                if (i >= 0 && i < By_staggered.size() && j < By_staggered[i].size())
                    By_staggered[i][j] = inflow_magnetic_field_y;

                // Set inflow boundary conditions for Bz_centered
                if (i >= 0 && i < Bz_centered.size() && j < Bz_centered[i].size())
                    Bz_centered[i][j] = inflow_magnetic_field_z;

                // Set inflow boundary conditions for Ex_staggered
                if (i >= 0 && i < Ex_staggered.size() && j < Ex_staggered[i].size())
                    Ex_staggered[i][j] = Egsm_x_1;

                // Set inflow boundary conditions for Ey_staggered
                if (i >= 0 && i < Ey_staggered.size() && j < Ey_staggered[i].size())
                    Ey_staggered[i][j] = Egsm_y_1;

                // Set inflow boundary conditions for ion density
                if (i >= 0 && i < ion_density_centered.size() && j < ion_density_centered[i].size())
                    ion_density_centered[i][j] = inflow_electron_density;

                // Set inflow boundary conditions for ion velocity x
                if (i >= 0 && i < ion_velocity_x_centered.size() && j < ion_velocity_x_centered[i].size())
                    ion_velocity_x_centered[i][j] = inflow_electron_velocity_x;

                // Set inflow boundary conditions for ion velocity y
                if (i >= 0 && i < ion_velocity_y_centered.size() && j < ion_velocity_y_centered[i].size())
                    ion_velocity_y_centered[i][j] = inflow_electron_velocity_y;
            }
        }
    }

    // Outflow Boundary Conditions (Right, Top, Bottom Boundaries)
    // Right boundary (i = local_NX - 1)
    for (int j = 0; j < local_NY; ++j) {
        int out_idx = local_NX - 1;

        // Right boundary extrapolation using 4 ghost cells
        for (int gc = 1; gc <= GHOST_CELLS; ++gc) {
            int i = out_idx + gc; // Index for the right boundary ghost cells

            // Apply extrapolation for Bx_staggered
            if (i < Bx_staggered.size() && j < Bx_staggered[i].size()) {
                Bx_staggered[i][j] = 4.0 * Bx_staggered[i - 1][j] - 6.0 * Bx_staggered[i - 2][j] 
                                     + 4.0 * Bx_staggered[i - 3][j] - Bx_staggered[i - 4][j];
            }

            // Apply extrapolation for Bz_centered
            if (i < Bz_centered.size() && j < Bz_centered[i].size()) {
                Bz_centered[i][j] = 4.0 * Bz_centered[i - 1][j] - 6.0 * Bz_centered[i - 2][j] 
                                     + 4.0 * Bz_centered[i - 3][j] - Bz_centered[i - 4][j];
            }

            // Apply extrapolation for Ex_staggered
            if (i < Ex_staggered.size() && j < Ex_staggered[i].size()) {
                Ex_staggered[i][j] = 4.0 * Ex_staggered[i - 1][j] - 6.0 * Ex_staggered[i - 2][j] 
                                     + 4.0 * Ex_staggered[i - 3][j] - Ex_staggered[i - 4][j];
            }

            // Apply extrapolation for ion_velocity_x_centered
            if (i < ion_velocity_x_centered.size() && j < ion_velocity_x_centered[i].size()) {
                ion_velocity_x_centered[i][j] = 4.0 * ion_velocity_x_centered[i - 1][j] - 6.0 * ion_velocity_x_centered[i - 2][j]
                                                + 4.0 * ion_velocity_x_centered[i - 3][j] - ion_velocity_x_centered[i - 4][j];
            }
        }
    }

    // Top boundary (j = local_NY - 1)
    for (int i = 0; i < local_NX; ++i) {
        for (int gc = 1; gc <= GHOST_CELLS; ++gc) {
            int j = local_NY - 1 + gc; // Index for the top boundary ghost cells

            // Apply extrapolation for By_staggered
            if (i < By_staggered.size() && j < By_staggered[i].size()) {
                By_staggered[i][j] = 4.0 * By_staggered[i][j - 1] - 6.0 * By_staggered[i][j - 2]
                                     + 4.0 * By_staggered[i][j - 3] - By_staggered[i][j - 4];
            }

            // Apply extrapolation for Ey_staggered
            if (i < Ey_staggered.size() && j < Ey_staggered[i].size()) {
                Ey_staggered[i][j] = 4.0 * Ey_staggered[i][j - 1] - 6.0 * Ey_staggered[i][j - 2]
                                     + 4.0 * Ey_staggered[i][j - 3] - Ey_staggered[i][j - 4];
            }
        }
    }

    // Bottom boundary (j = 0)
    for (int i = 0; i < local_NX; ++i) {
        for (int gc = 1; gc <= GHOST_CELLS; ++gc) {
            int j =  - gc; // Index for the bottom boundary ghost cells

            // Apply extrapolation for By_staggered
            if (i >= 0 && i < By_staggered.size() && j >= 0 && j < By_staggered[i].size()) {
                By_staggered[i][j] = 4.0 * By_staggered[i][j + 1] - 6.0 * By_staggered[i][j + 2]
                                    + 4.0 * By_staggered[i][j + 3] - By_staggered[i][j + 4];
            }

            // Apply extrapolation for Ey_staggered
            if (i >= 0 && i < Ey_staggered.size() && j >= 0 && j < Ey_staggered[i].size()) {
                Ey_staggered[i][j] = 4.0 * Ey_staggered[i][j + 1] - 6.0 * Ey_staggered[i][j + 2]
                                    + 4.0 * Ey_staggered[i][j + 3] - Ey_staggered[i][j + 4];
            }
        }
    }

    // MPI Communication for Ghost Cells
    // rank 0 sends to the last rank and receives from rank 1
    if (rank == 0) {
        // Send to rank 1 (next)
        for (int j = 0; j < local_NY; ++j) {
            MPI_Isend(&Bz_centered[GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(&Bx_staggered[GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &requests[4]);
            MPI_Isend(&Ex_staggered[GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &requests[8]);
        }
        for (int i = 0; i < local_NX; ++i) {
            MPI_Isend(&By_staggered[i][GHOST_CELLS], 1, MPI_DOUBLE, 1, 8, MPI_COMM_WORLD, &requests[16]);
            MPI_Isend(&Ey_staggered[i][GHOST_CELLS], 1, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &requests[20]);
        }
    // Receive from rank 1 (next)
        for (int j = 0; j < local_NY; ++j) {
            MPI_Irecv(&Bz_centered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(&Bx_staggered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &requests[5]);
            MPI_Irecv(&Ex_staggered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &requests[9]);
        }
        for (int i = 0; i < local_NX; ++i) {
            MPI_Irecv(&By_staggered[i][local_NY + GHOST_CELLS], 1, MPI_DOUBLE, 1, 8, MPI_COMM_WORLD, &requests[17]);
            MPI_Irecv(&Ey_staggered[i][local_NY + GHOST_CELLS], 1, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &requests[21]);
        }
    }

    // rank size-1 receives from the second to last rank and sends to rank 0
    else if (rank == size - 1) {
        // Send to rank 0 (prev)
        for (int j = 0; j < local_NY; ++j) {
            MPI_Isend(&Bz_centered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(&Bx_staggered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[6]);
            MPI_Isend(&Ex_staggered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, &requests[10]);
        }
        for (int i = 0; i < local_NX; ++i) {
            MPI_Isend(&By_staggered[i][local_NY + GHOST_CELLS - 1], 1, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD, &requests[18]);
            MPI_Isend(&Ey_staggered[i][local_NY + GHOST_CELLS - 1], 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &requests[22]);
        }
    // Receive from rank 0 (prev)
        for (int j = 0; j < local_NY; ++j) {
            MPI_Irecv(&Bz_centered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[3]);
            MPI_Irecv(&Bx_staggered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[7]);
            MPI_Irecv(&Ex_staggered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, &requests[11]);
        }
        for (int i = 0; i < local_NX; ++i) {
            MPI_Irecv(&By_staggered[i][GHOST_CELLS - 1], 1, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD, &requests[19]);
            MPI_Irecv(&Ey_staggered[i][GHOST_CELLS - 1], 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, &requests[23]);
        }
    }

    // all other ranks send to next rank and receive from the previous rank
    else if (rank != MPI_PROC_NULL) {
        // Send to next rank
        for (int j = 0; j < local_NY; ++j) {
            MPI_Isend(&Bz_centered[GHOST_CELLS][j], 1, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(&Bz_centered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(&Bx_staggered[GHOST_CELLS][j], 1, MPI_DOUBLE, prev_rank, 2, MPI_COMM_WORLD, &requests[4]);
            MPI_Isend(&Bx_staggered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, next_rank, 3, MPI_COMM_WORLD, &requests[6]);
            MPI_Isend(&Ex_staggered[GHOST_CELLS][j], 1, MPI_DOUBLE, prev_rank, 4, MPI_COMM_WORLD, &requests[8]);
            MPI_Isend(&Ex_staggered[local_NX + GHOST_CELLS - 1][j], 1, MPI_DOUBLE, next_rank, 5, MPI_COMM_WORLD, &requests[10]);
        }
        for (int i = 0; i < local_NX; ++i) {
            MPI_Isend(&By_staggered[i][GHOST_CELLS], 1, MPI_DOUBLE, prev_rank, 8, MPI_COMM_WORLD, &requests[16]);
            MPI_Isend(&By_staggered[i][local_NY + GHOST_CELLS - 1], 1, MPI_DOUBLE, next_rank, 9, MPI_COMM_WORLD, &requests[18]);
            MPI_Isend(&Ey_staggered[i][GHOST_CELLS], 1, MPI_DOUBLE, prev_rank, 10, MPI_COMM_WORLD, &requests[20]);
            MPI_Isend(&Ey_staggered[i][local_NY + GHOST_CELLS - 1], 1, MPI_DOUBLE, next_rank, 11, MPI_COMM_WORLD, &requests[22]);
        }

        // Receive from previous rank
        for (int j = 0; j < local_NY; ++j) {
            MPI_Irecv(&Bz_centered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &requests[3]);
            MPI_Irecv(&Bz_centered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(&Bx_staggered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, prev_rank, 3, MPI_COMM_WORLD, &requests[7]);
            MPI_Irecv(&Bx_staggered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, next_rank, 2, MPI_COMM_WORLD, &requests[5]);
            MPI_Irecv(&Ex_staggered[GHOST_CELLS - 1][j], 1, MPI_DOUBLE, prev_rank, 5, MPI_COMM_WORLD, &requests[11]);
            MPI_Irecv(&Ex_staggered[local_NX + GHOST_CELLS][j], 1, MPI_DOUBLE, next_rank, 4, MPI_COMM_WORLD, &requests[9]);
        }

        for (int i = 0; i < local_NX; ++i) {
            MPI_Irecv(&By_staggered[i][GHOST_CELLS - 1], 1, MPI_DOUBLE, prev_rank, 9, MPI_COMM_WORLD, &requests[19]);
            MPI_Irecv(&By_staggered[i][local_NY + GHOST_CELLS], 1, MPI_DOUBLE, next_rank, 8, MPI_COMM_WORLD, &requests[17]);
            MPI_Irecv(&Ey_staggered[i][GHOST_CELLS - 1], 1, MPI_DOUBLE, prev_rank, 11, MPI_COMM_WORLD, &requests[23]);
            MPI_Irecv(&Ey_staggered[i][local_NY + GHOST_CELLS], 1, MPI_DOUBLE, next_rank, 10, MPI_COMM_WORLD, &requests[21]);
        }
    }

    MPI_Waitall(24, requests, statuses); 
}

void initializeFieldsAndParticles() {
    initializeElectricFields(Ex_staggered, Ey_staggered, local_NX, local_NY); // Initialize electric fields on staggered grid
    initializeFluids();         // Initialize fluid properties using centered variables
    initializeParticles();      // Initialize particles
    initializeElectronFields(); // Initialize electron properties using centered variables

    // Synchronize all fields in a batch using the staggered versions
    std::vector<std::vector<std::vector<double>>> fields_to_sync = {
        Ex_staggered, Ey_staggered, 
        ion_density_centered, ion_pressure_centered,
        electron_velocity_x, electron_velocity_y, electron_pressure_centered
    };
    std::vector<std::string> field_names = {
        "Ex_staggered", "Ey_staggered",
        "ion_density_centered", "ion_pressure_centered",
        "electron_velocity_x", "electron_velocity_y", "electron_pressure_centered"
    };
    exchangeGhostCellsBatchMPI(fields_to_sync, field_names, local_NX, local_NY, rank, size);

    // Debug output to confirm initialization
    if (rank == 0) {
        std::cout << "Rank " << rank << ": Fields and particles initialized and synchronized." << std::endl;
    }
}

// Electricfields CT
void updateElectricFieldCT(
    std::vector<std::vector<double>>& Ex_staggered,
    std::vector<std::vector<double>>& Ey_staggered,
    const std::vector<std::vector<double>>& Bx_staggered,
    const std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& Bz_centered,
    const std::vector<std::vector<double>>& vx,
    const std::vector<std::vector<double>>& vy,
    const std::vector<std::vector<double>>& electron_pressure_centered,
    const std::vector<std::vector<double>>& ion_density_centered,
    const std::vector<std::vector<double>>& Jx_staggered,
    const std::vector<std::vector<double>>& Jy_staggered,
    int local_NX, int local_NY)
{
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS + 1; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            if (i >= Ex_staggered.size() || j >= Ex_staggered[0].size()) continue;
            double grad_Pe_x = 0.0;
            // Compute electron pressure gradients using fourth-order schemes
            // Gradient in the x-direction, Ex_staggered is at (i+1/2, j)
            if (i < GHOST_CELLS + 3) { // Near the left boundary
                grad_Pe_x = (-25.0 * electron_pressure_centered[i][j] + 48.0 * electron_pressure_centered[i + 1][j]
                             - 36.0 * electron_pressure_centered[i + 2][j] + 16.0 * electron_pressure_centered[i + 3][j]
                             - 3.0 * electron_pressure_centered[i + 4][j]) / (12.0 * DX);
            } else if (i > local_NX + GHOST_CELLS - 4) { // Near the right boundary
                grad_Pe_x = (25.0 * electron_pressure_centered[i][j] - 48.0 * electron_pressure_centered[i - 1][j]
                             + 36.0 * electron_pressure_centered[i - 2][j] - 16.0 * electron_pressure_centered[i - 3][j]
                             + 3.0 * electron_pressure_centered[i - 4][j]) / (12.0 * DX);
            } else { // Interior
                grad_Pe_x = (-electron_pressure_centered[i + 2][j] + 8.0 * electron_pressure_centered[i + 1][j]
                             - 8.0 * electron_pressure_centered[i - 1][j] + electron_pressure_centered[i - 2][j]) / (12.0 * DX);
            }

            // Compute Hall term for Ex_staggered
            double hall_term_x = 0.0;
            if (j > 0 && j < local_NY + 2 * GHOST_CELLS) {
                // Interpolate Jy to (i+1/2, j)
                double Jy_interp = (i < local_NX - 1 && j < Jy_staggered[0].size() - 1) ? 0.5 * (Jy_staggered[i][j] + Jy_staggered[i + 1][j]) : 0.0;
                // Interpolate By to (i+1/2, j)
                double By_interp = (i < local_NX - 1 && j < By_staggered[0].size() - 1) ? 0.5 * (By_staggered[i][j] + By_staggered[i + 1][j]) : By_staggered[i][j];
                // Bz is at (i, j), use directly
                double Bz_val = (i < local_NX - 1) ? 0.5 * (Bz_centered[i][j] + Bz_centered[i + 1][j]) : 0.0;

                hall_term_x = (Jy_interp * Bz_val - Jx_staggered[i][j] * By_interp) / (ion_density_centered[i][j] * QE);
            }
            
            // Interpolate vy to (i+1/2, j) for v_cross_B_x calculation
            double vy_interp = (i < local_NX - 1) ? 0.5 * (vy[i][j] + vy[i + 1][j]) : vy[i][j];
            // Compute v_cross_B_x at (i+1/2, j)
            double v_cross_B_x = vy_interp * Bz_centered[i][j] - vx[i][j] * 0.5 * (By_staggered[i][j] + By_staggered[i + 1][j]); // Assuming vx is at cell center

            // Update Ex_staggered
            Ex_staggered[i][j] = -v_cross_B_x - grad_Pe_x + hall_term_x;
        }
    }

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS + 1; ++j) {
            if (i >= Ey_staggered.size() || j >= Ey_staggered[0].size()) continue;
            double grad_Pe_y = 0.0;
            // Compute electron pressure gradients using fourth-order schemes
            // Gradient in the y-direction, Ey_staggered is at (i, j+1/2)
            if (j < GHOST_CELLS + 3) { // Near the bottom boundary
                grad_Pe_y = (-25.0 * electron_pressure_centered[i][j] + 48.0 * electron_pressure_centered[i][j + 1]
                             - 36.0 * electron_pressure_centered[i][j + 2] + 16.0 * electron_pressure_centered[i][j + 3]
                             - 3.0 * electron_pressure_centered[i][j + 4]) / (12.0 * DY);
            } else if (j > local_NY + GHOST_CELLS - 4) { // Near the top boundary
                grad_Pe_y = (25.0 * electron_pressure_centered[i][j] - 48.0 * electron_pressure_centered[i][j - 1]
                             + 36.0 * electron_pressure_centered[i][j - 2] - 16.0 * electron_pressure_centered[i][j - 3]
                             + 3.0 * electron_pressure_centered[i][j - 4]) / (12.0 * DY);
            } else { // Interior
                grad_Pe_y = (-electron_pressure_centered[i][j + 2] + 8.0 * electron_pressure_centered[i][j + 1]
                             - 8.0 * electron_pressure_centered[i][j - 1] + electron_pressure_centered[i][j - 2]) / (12.0 * DY);
            }

            // Compute Hall term for Ey_staggered
            double hall_term_y = 0.0;
            if (i > 0 && i < local_NX) {
                // Interpolate Jx to (i, j+1/2)
                double Jx_interp = (j < local_NY - 1 && i < Jx_staggered.size() - 1) ? 0.5 * (Jx_staggered[i][j] + Jx_staggered[i][j + 1]) : 0.0;
                // Interpolate Bx to (i, j+1/2)
                double Bx_interp = (j < local_NY - 1 && i < Bx_staggered.size() - 1) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i][j + 1]) : Bx_staggered[i][j];
                // Bz is at (i, j), use directly
                double Bz_val = (j < local_NY - 1) ? 0.5 * (Bz_centered[i][j] + Bz_centered[i][j + 1]) : 0.0;

                hall_term_y = (Jx_interp * Bz_val - Jy_staggered[i][j] * Bx_interp) / (ion_density_centered[i][j] * QE);
            }

            // Interpolate vx to (i, j+1/2) for v_cross_B_y calculation
            double vx_interp = (j < local_NY - 1) ? 0.5 * (vx[i][j] + vx[i][j + 1]) : vx[i][j];
            // Compute v_cross_B_y at (i, j+1/2)
            double v_cross_B_y = vx_interp * Bz_centered[i][j] - vy[i][j] * 0.5 * (Bx_staggered[i][j] + Bx_staggered[i][j + 1]); // Assuming vy is at cell center

            // Update Ey_staggered
            Ey_staggered[i][j] = -v_cross_B_y - grad_Pe_y + hall_term_y;
        }
    }
}
//Initialize grid structure
void initializeCTGridStructure(int local_NX, int local_NY) {
    // Sanity check for dimensions
    if (local_NX <= 0 || local_NY <= 0 || GHOST_CELLS < 0) {
        std::cerr << "Error on rank " << rank << ": Invalid dimensions in initializeCTGridStructure. "
                  << "local_NX = " << local_NX << ", local_NY = " << local_NY
                  << ", GHOST_CELLS = " << GHOST_CELLS << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    try {
        // Debugging: Print resizing parameters
        std::cout << "Rank " << rank << ": Resizing arrays with local_NX = "
                  << local_NX << ", local_NY = " << local_NY
                  << ", GHOST_CELLS = " << GHOST_CELLS << std::endl;

        // Resize staggered electric fields
        Ex_staggered.resize(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        Ey_staggered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0));

        // Resize staggered magnetic fields
        Bx_staggered.resize(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        By_staggered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0));
        Bz_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

        // Resize plasma properties with correct ghost cell dimensions
        ion_density_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        ion_pressure_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        ion_velocity_x_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        ion_velocity_y_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        electron_velocity_x_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        electron_velocity_y_centered.resize(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error on rank " << rank << ": Memory allocation failed in initializeCTGridStructure. "
                  << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Debugging output for sizes
    std::cout << "Rank " << rank << ": Sizes initialized in initializeCTGridStructure:" << std::endl;
    std::cout << "ion_density_centered: " << ion_density_centered.size() << " x "
              << (ion_density_centered.empty() ? 0 : ion_density_centered[0].size()) << std::endl;

    // Assertions to validate allocation
    assert(!ion_density_centered.empty());
    assert(ion_density_centered.size() == local_NX);
    assert(ion_density_centered[0].size() == local_NY + 2 * GHOST_CELLS);

    std::cout << "Rank " << rank << ": Grid structure initialized successfully with ghost cells." << std::endl;
}

//inflow velocity
double calculateInflowVelocityMPI(
    const std::vector<std::vector<double>>& ion_velocity_x_centered,
    const std::vector<std::vector<double>>& ion_velocity_y_centered,
    int global_x_point_i, int global_x_point_j, int local_NX, int local_NY, int rank, int size
) {
    // Initialize inflow velocity to zero
    double inflow_velocity = 0.0;

    // Convert global indices to local indices for this rank
    int local_x_point_i = global_x_point_i - (rank % px) * (NX / px);
    int local_x_point_j = global_x_point_j - (rank / px) * (NY / py);

    // Check if the X-point lies within the local subdomain
    if (local_x_point_i >= GHOST_CELLS && local_x_point_i < local_NX - GHOST_CELLS &&
        local_x_point_j >= GHOST_CELLS && local_x_point_j < local_NY + GHOST_CELLS) {

        // Calculate inflow velocity using centered values
        double vx = ion_velocity_x_centered[local_x_point_i][local_x_point_j];
        double vy = ion_velocity_y_centered[local_x_point_i][local_x_point_j];

        inflow_velocity = std::sqrt(vx * vx + vy * vy);
    }

    // Ensure all processes have the same inflow velocity
    MPI_Bcast(&inflow_velocity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return inflow_velocity;
}

// Compute Derivatives with MPI
void computeDerivativesMPI(
    const std::vector<std::vector<double>>& phi_centered, // Input potential at cell centers
    std::vector<std::vector<double>>& dPhi_dx_staggered, // Output x-derivative at x-faces
    std::vector<std::vector<double>>& dPhi_dy_staggered, // Output y-derivative at y-faces
    int local_NX, int local_NY, int rank, int size
) {

    // Exchange ghost cells for phi_centered before computing derivatives
    exchangeGhostCellsMPI(const_cast<std::vector<std::vector<double>>&>(phi_centered), "phi_centered", local_NX, local_NY, rank, size);

    // Compute dPhi_dx at x-faces (i, j)
    for (int i = GHOST_CELLS; i < local_NX + 1 - GHOST_CELLS; ++i) { // Corrected loop bound for x-faces
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            if (i < dPhi_dx_staggered.size() && j < dPhi_dx_staggered[0].size()) {
                dPhi_dx_staggered[i][j] = (phi_centered[i][j] - phi_centered[i - 1][j]) / DX; // (i,j) is the x-face between (i-1/2,j) and (i+1/2,j)
            }
        }
    }

    // Compute dPhi_dy at y-faces (i, j)
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + 1; ++j) { // Corrected loop bound for y-faces
            if (i < dPhi_dy_staggered.size() && j < dPhi_dy_staggered[0].size()) {
                dPhi_dy_staggered[i][j] = (phi_centered[i][j] - phi_centered[i][j - 1]) / DY; // (i,j) is the y-face between (i,j-1/2) and (i,j+1/2)
            }
        }
    }

    // Exchange ghost cells for the computed derivatives
    exchangeGhostCellsMPI(dPhi_dx_staggered, "dPhi_dx_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(dPhi_dy_staggered, "dPhi_dy_staggered", local_NX, local_NY, rank, size);
}

//Boris correction
void applyBorisCorrectionMPI(
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& phi_B_centered,
    int local_NX, int local_NY, int rank, int size
) {
    // Local derivatives at the correct staggered locations
    std::vector<std::vector<double>> dPhi_dx_staggered(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0)); // For Bx_staggered
    std::vector<std::vector<double>> dPhi_dy_staggered(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0)); // For By_staggered

    // Compute derivatives of phi_B at cell centers
    computeDerivativesMPI(phi_B_centered, dPhi_dx_staggered, dPhi_dy_staggered, local_NX, local_NY, rank, size);

    // Bx correction at x-faces (i, j)
    for (int i = 0; i < Bx_staggered.size(); ++i) {
        for (int j = 0; j < Bx_staggered[0].size(); ++j) {
            Bx_staggered[i][j] -= dPhi_dx_staggered[i][j]; // No +0.5 needed for i since Bx is at x-faces
        }
    }

    // By correction at y-faces (i, j)
    for (int i = 0; i < By_staggered.size(); ++i) {
        for (int j = 0; j < By_staggered[0].size(); ++j) {
            By_staggered[i][j] -= dPhi_dy_staggered[i][j]; // No +0.5 needed for j since By is at y-faces
        }
    }

    // Exchange ghost cells for Bx_staggered and By_staggered after updating
    exchangeGhostCellsMPI(Bx_staggered, "Bx_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(By_staggered, "By_staggered", local_NX, local_NY, rank, size);
}

//Magnetic Divergence
double computeMagneticDivergenceMPI(
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& Bz_centered,
    std::vector<std::vector<double>>& div_B,
    int local_NX, int local_NY, int rank, int size) {
    
    // Local derivatives with ghost cells
    std::vector<std::vector<double>> dBx_dx(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> dBy_dy(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

    // Exchange ghost cells for Bx_staggered and By_staggered before computing derivatives
    exchangeGhostCellsMPI(Bx_staggered, "Bx_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(By_staggered, "By_staggered", local_NX, local_NY, rank, size);

    // Compute derivatives dBx_dx and dBy_dy using the updated Bx_staggered and By_staggered
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Calculate dBx_dx at cell center (i, j) using Bx_staggered at (i+1/2, j) and (i-1/2, j)
            if (i + 1 < Bx_staggered.size() && i > 0) {
                dBx_dx[i][j] = (Bx_staggered[i + 1][j] - Bx_staggered[i][j]) / DX;
            }

            // Calculate dBy_dy at cell center (i, j) using By_staggered at (i, j+1/2) and (i, j-1/2)
            if (j + 1 < By_staggered[i].size() && j > 0) {
                dBy_dy[i][j] = (By_staggered[i][j + 1] - By_staggered[i][j]) / DY;
            }
        }
    }

    // Compute divergence locally at cell centers
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            div_B[i][j] = dBx_dx[i][j] + dBy_dy[i][j];
        }
    }

    // Synchronize ghost cells for div_B
    exchangeGhostCellsMPI(div_B, "div_B", local_NX, local_NY, rank, size);

    // Compute local maximum absolute divergence
    double local_max_div = 0.0;
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            local_max_div = std::max(local_max_div, std::abs(div_B[i][j]));
        }
    }

    // Reduce to find the global maximum divergence
    double global_max_div = 0.0;
    MPI_Reduce(&local_max_div, &global_max_div, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Global Maximum Magnetic Divergence: " << global_max_div << std::endl;
    }

    return global_max_div;
}

std::vector<std::vector<double>> computeMagneticScalarPotentialMPI(
    const std::vector<std::vector<double>>& local_div_B, int local_NX, int local_NY, int levels, int rank, int size) {

    // Solve the Poisson equation using the multigrid solver locally
    std::vector<std::vector<double>> local_phi = solvePoissonMultigridMPI(local_div_B, local_NX, local_NY, levels, "phi_B", rank, size);

    // Return the local potential with ghost cells for further computations
    return local_phi;
}

std::pair<int, int> findXPointFromPotentialMPI(const std::vector<std::vector<double>>& local_phi) {
    int local_x_point_i = -1, local_x_point_j = -1;
    double local_min_curvature = std::numeric_limits<double>::max();

    // Compute the X-point locally
    for (int i = GHOST_CELLS + 1; i < local_NX - GHOST_CELLS - 1; ++i) {
        for (int j = GHOST_CELLS + 1; j < local_NY - GHOST_CELLS - 1; ++j) {
            double d2phi_dx2 = local_phi[i + 1][j] - 2.0 * local_phi[i][j] + local_phi[i - 1][j];
            double d2phi_dy2 = local_phi[i][j + 1] - 2.0 * local_phi[i][j] + local_phi[i][j - 1];
            double curvature = std::abs(d2phi_dx2 + d2phi_dy2);

            if (curvature < local_min_curvature) {
                local_min_curvature = curvature;
                local_x_point_i = i;
                local_x_point_j = j;
            }
        }
    }

    // Find the global minimum curvature and corresponding rank
    double global_min_curvature;
    int global_min_rank;
    int global_x_point_i;
    int global_x_point_j;

    // Structure to hold curvature and rank for MPI reduction
    struct {
        double val;
        int   rank;
    } local_min = {local_min_curvature, rank}, global_min;

    // Use MPI_Allreduce to find the global minimum and its location (rank)
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    // Update global_x_point_i and global_x_point_j based on the result from the rank with the minimum curvature
    if (rank == global_min.rank) {
        global_x_point_i = local_x_point_i;
        global_x_point_j = local_x_point_j;
    }

    // Broadcast the global x-point coordinates from the rank with the minimum curvature to all other ranks
    MPI_Bcast(&global_x_point_i, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
    MPI_Bcast(&global_x_point_j, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);

    // Convert local indices to global indices
    int global_i = (rank % px) * (NX / px) + global_x_point_i - GHOST_CELLS;
    int global_j = (rank / px) * (NY / py) + global_x_point_j - GHOST_CELLS;

    return {global_i, global_j};
}
// Total Current 
void calculateTotalCurrentMPI(
    std::vector<std::vector<double>>& Jx_staggered,
    std::vector<std::vector<double>>& Jy_staggered,
    std::vector<std::vector<double>>& Jz_centered,
    const std::vector<std::vector<double>>& Bz_centered,
    const std::vector<std::vector<double>>& Bx_staggered,
    const std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& ion_density_centered
) {
    // Calculate Jx on x-faces (i+1/2, j)
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS + 1; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            if (i < Jx_staggered.size() && j < Jx_staggered[i].size()) {
                // Interpolate Bz to (i+1/2, j) for dBz_dy calculation
                double Bz_at_x_face = (j > 0 && j < Bz_centered[0].size() - 1) ? 0.5 * (Bz_centered[i][j] + Bz_centered[i][j - 1]) : 0.0;
                double dBz_dy = (j > 0 && j < Bz_centered[0].size() - 1) ? (Bz_centered[i][j] - Bz_centered[i][j - 1]) / DY : 0.0;
                Jx_staggered[i][j] = dBz_dy / MU_0;
            }
        }
    }

    // Calculate Jy on y-faces (i, j+1/2)
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            if (i < Jy_staggered.size() && j < Jy_staggered[i].size()) {
                // Interpolate Bz to (i, j+1/2) for dBz_dx calculation
                double Bz_at_y_face = (i > 0 && i < Bz_centered.size() - 1) ? 0.5 * (Bz_centered[i][j] + Bz_centered[i - 1][j]) : 0.0;
                double dBz_dx = (i > 0 && i < Bz_centered.size() - 1) ? (Bz_centered[i][j] - Bz_centered[i - 1][j]) / DX : 0.0;
                Jy_staggered[i][j] = -dBz_dx / MU_0;
            }
        }
    }

    // Calculate Jz at cell centers (i, j)
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Interpolate By to (i+1/2, j) for dBy_dx calculation
            double By_at_x_face = (i < By_staggered.size() - 1) ? 0.5 * (By_staggered[i][j] + By_staggered[i + 1][j]) : 0.0;
            double dBy_dx = (i < local_NX - 1) ? (By_at_x_face - By_staggered[i][j]) / DX : 0.0;

            // Interpolate Bx to (i, j+1/2) for dBx_dy calculation
            double Bx_at_y_face = (j < Bx_staggered[0].size() - 1) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i][j + 1]) : 0.0;
            double dBx_dy = (j < local_NY - 1) ? (Bx_at_y_face - Bx_staggered[i][j]) / DY : 0.0;

            Jz_centered[i][j] = (dBy_dx - dBx_dy) / MU_0;
        }
    }
}

// WENO weights
std::vector<WENOWeights> precomputeWENOWeightsMPI(int num_weights) {
    const double epsilon = 1e-6;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (num_weights < size) {
        if (rank == 0) {
            std::cerr << "Error: Number of weights must be >= number of processes." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int weights_per_process = num_weights / size;
    int remainder = num_weights % size;
    int start_index = rank * weights_per_process + std::min(rank, remainder);
    int end_index = start_index + weights_per_process + (rank < remainder ? 1 : 0);

    std::vector<WENOWeights> local_weights(end_index - start_index);
    for (int i = start_index; i < end_index; ++i) {
        double beta0 = std::pow(i - 2, 2); // These are smoothness indicators
        double beta1 = std::pow(i - 1, 2);
        double beta2 = std::pow(i, 2);

        double alpha0 = 0.1 / std::pow(epsilon + beta0, 2); // These are the WENO weights
        double alpha1 = 0.6 / std::pow(epsilon + beta1, 2);
        double alpha2 = 0.3 / std::pow(epsilon + beta2, 2);

        double sum_alpha = alpha0 + alpha1 + alpha2;
        if (sum_alpha > 1e-30) {
            // Normalize the weights
            local_weights[i - start_index].alpha0 = alpha0 / sum_alpha;
            local_weights[i - start_index].alpha1 = alpha1 / sum_alpha;
            local_weights[i - start_index].alpha2 = alpha2 / sum_alpha;
        } else {
            // Handle cases where sum_alpha is too small
            local_weights[i - start_index].alpha0 = 0.0;
            local_weights[i - start_index].alpha1 = 0.0;
            local_weights[i - start_index].alpha2 = 0.0;
        }
    }

    return local_weights;
}

// WENO Application
void applyWENO(
    std::vector<std::vector<double>>& field,
    const std::vector<WENOWeights>& weights,
    const std::string& field_type
) {
    // Temporary field to store the updated values
    std::vector<std::vector<double>> temp_field = field;

    // Ensure weights size matches the grid
    if (weights.size() < (local_NX - 2 * GHOST_CELLS)) {
        throw std::runtime_error("Insufficient WENO weights for the grid.");
    }

    // Ensure ghost cells are up-to-date
    exchangeGhostCellsMPI(field, field_type, local_NX, local_NY, rank, size);

    // Local computation with boundary handling
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Handle boundaries explicitly using lower-order schemes
            if (i - 2 < GHOST_CELLS || i + 2 >= local_NX - GHOST_CELLS ||
                j - 2 < GHOST_CELLS || j + 2 >= local_NY + GHOST_CELLS) {

                // Near the left boundary (i < GHOST_CELLS + 2)
                if (i < GHOST_CELLS + 2) {
                    // Use lower-order WENO or one-sided difference
                    // Example: 2nd order one-sided for simplicity
                    temp_field[i][j] = (-3.0 * field[i][j] + 4.0 * field[i + 1][j] - field[i + 2][j]) / (2.0 * DX);
                }
                // Near the right boundary (i >= local_NX - GHOST_CELLS - 2)
                else if (i >= local_NX - GHOST_CELLS - 2) {
                    // Use lower-order WENO or one-sided difference
                    // Example: 2nd order one-sided for simplicity
                    temp_field[i][j] = (3.0 * field[i][j] - 4.0 * field[i - 1][j] + field[i - 2][j]) / (2.0 * DX);
                }
                // Near the bottom boundary (j < GHOST_CELLS + 2)
                else if (j < GHOST_CELLS + 2) {
                    // Use lower-order WENO or one-sided difference
                    // Example: 2nd order one-sided for simplicity
                    temp_field[i][j] = (-3.0 * field[i][j] + 4.0 * field[i][j + 1] - field[i][j + 2]) / (2.0 * DY);
                }
                // Near the top boundary (j >= local_NY + GHOST_CELLS - 2)
                else if (j >= local_NY + GHOST_CELLS - 2) {
                    // Use lower-order WENO or one-sided difference
                    // Example: 2nd order one-sided for simplicity
                    temp_field[i][j] = (3.0 * field[i][j] - 4.0 * field[i][j - 1] + field[i][j - 2]) / (2.0 * DY);
                }

                continue; // Skip to next cell after applying boundary scheme
            }

            // WENO stencil: Collect the required points
            std::vector<double> stencil_x = {
                field[i - 2][j], field[i - 1][j], field[i][j],
                field[i + 1][j], field[i + 2][j]
            };

            // Validate indexing for weights
            const auto& w = weights[i - GHOST_CELLS];

            // WENO reconstruction
            double P0 = (2 * stencil_x[0] - 7 * stencil_x[1] + 11 * stencil_x[2]) / 6.0;
            double P1 = (-stencil_x[1] + 5 * stencil_x[2] + 2 * stencil_x[3]) / 6.0;
            double P2 = (2 * stencil_x[2] + 5 * stencil_x[3] - stencil_x[4]) / 6.0;

            temp_field[i][j] = w.alpha0 * P0 + w.alpha1 * P1 + w.alpha2 * P2;
        }
    }

    // Update the field with the local computation
    field = temp_field;

    // Synchronize ghost cells across processes
    exchangeGhostCellsMPI(field, field_type, local_NX, local_NY, rank, size);
}

// Update electromagnetic fields using FDTD with MPI
void updateFields(
    std::vector<std::vector<double>>& Ex_staggered,
    std::vector<std::vector<double>>& Ey_staggered,
    std::vector<std::vector<double>>& Bz_centered,
    const std::vector<std::vector<double>>& Jx_staggered,
    const std::vector<std::vector<double>>& Jy_staggered,
    std::vector<std::vector<double>>& Bx_staggered,
    std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& vx,
    const std::vector<std::vector<double>>& vy,
    const std::vector<std::vector<double>>& electron_pressure_centered,
    int local_NX, int local_NY, int rank, int size) {

    // Step 1: Update Ex_staggered and Ey_staggered based on Jx_staggered and Jy_staggered
    for (int i = GHOST_CELLS; i < local_NX + 1; ++i) { // Note: Ex_staggered has one more element in the x-direction
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            if (i < Ex_staggered.size() && j < Ex_staggered[i].size() && i < Jy_staggered.size() && j < Jy_staggered[i].size()) {
                Ex_staggered[i][j] += DT / EPS_0 * Jy_staggered[i][j];
            }
        }
    }

    for (int i = GHOST_CELLS; i < local_NX; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS + 1; ++j) { // Note: Ey_staggered has one more element in the y-direction
            if (i < Ey_staggered.size() && j < Ey_staggered[i].size() && i < Jx_staggered.size() && j < Jx_staggered[i].size()) {
                Ey_staggered[i][j] -= DT / EPS_0 * Jx_staggered[i][j];
            }
        }
    }

    // Step 2: Calculate the divergence of electron pressure using electron_pressure_centered
    std::vector<std::vector<double>> grad_Pe_x(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> grad_Pe_y(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            grad_Pe_x[i][j] = (electron_pressure_centered[i + 1][j] - electron_pressure_centered[i - 1][j]) / (2.0 * DX);
            grad_Pe_y[i][j] = (electron_pressure_centered[i][j + 1] - electron_pressure_centered[i][j - 1]) / (2.0 * DY);
        }
    }

    // Batch of fields to synchronize ghost cells for updateElectricFieldCT and computeStaggeredCurl
    std::vector<std::vector<std::vector<double>>> fields_to_sync_batch = {
        {Ex_staggered, Ey_staggered, Bz_centered, Bx_staggered, By_staggered, electron_pressure_centered}
    };

    std::vector<std::string> field_types_update = {
        "Ex_staggered", "Ey_staggered", "Bz_centered", "Bx_staggered", "By_staggered", "electron_pressure_centered"
    };

    // Initiate batched ghost cell exchange
    exchangeGhostCellsBatchMPI(fields_to_sync_batch, field_types_update, local_NX, local_NY, rank, size);

    // Step 4: Update Electric Fields using CT (ghost cell dependent)
    updateElectricFieldCT(Ex_staggered, Ey_staggered, Bx_staggered, By_staggered, Bz_centered, vx, vy, electron_pressure_centered, ion_density_centered, Jx_staggered, Jy_staggered, local_NX, local_NY);

    // Step 5: Compute Magnetic Field (Bz) using Staggered Curl (ghost cell dependent)
    computeStaggeredCurl(Ex_staggered, Ey_staggered, Bz_centered, ion_density_centered, Jx_staggered, Jy_staggered, Bx_staggered, By_staggered, local_NX, local_NY, rank, size);

    // Step 6: Apply staggered boundary conditions after updates
    applyStaggeredBoundaryConditionsMPI(Bx_staggered, By_staggered, Bz_centered, Ex_staggered, Ey_staggered, ion_density_centered, ion_velocity_x_centered, ion_velocity_y_centered, local_NX, local_NY, rank, size);

    if (rank == 0) {
        std::cout << "Fields updated with CT method, overlapping communication and computation on rank "
                  << rank << "." << std::endl;
    }
}

// Update Electron Velocity based on Hall with MPI
void updateElectronVelocity(
    std::vector<std::vector<double>>& electron_velocity_x,
    std::vector<std::vector<double>>& electron_velocity_y,
    std::vector<std::vector<double>>& Ex_staggered,
    std::vector<std::vector<double>>& Ey_staggered,
    std::vector<std::vector<double>>& Bz_centered,
    std::vector<std::vector<double>>& electron_pressure_centered,
    std::vector<std::vector<double>>& ion_density_centered,
    int local_NX, int local_NY, int rank, int size
) {
    // Exchange ghost cells for required fields
    exchangeGhostCellsMPI(Ex_staggered, "Ex_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Ey_staggered, "Ey_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Bz_centered, "Bz_centered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(electron_pressure_centered, "electron_pressure_centered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_density_centered, "ion_density_centered", local_NX, local_NY, rank, size);

    // Update electron velocities in the local domain
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Ensure ion density is non-zero to avoid division by zero
            double density = std::max(ion_density_centered[i][j], 1e-10);

            // Calculate pressure gradients at cell-centers using central differences
            double grad_pressure_x = (electron_pressure_centered[i + 1][j] - electron_pressure_centered[i - 1][j]) / (2.0 * DX);
            double grad_pressure_y = (electron_pressure_centered[i][j + 1] - electron_pressure_centered[i][j - 1]) / (2.0 * DY);

            // Interpolate Ex and Ey to cell centers (i, j)
            double Ex_val = (i < Ex_staggered.size() - 1) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i + 1][j]) : Ex_staggered[i][j];
            double Ey_val = (j < Ey_staggered[0].size() - 1) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j + 1]) : Ey_staggered[i][j];

            // Bz is already at cell-center, so no interpolation needed
            double Bz_val = Bz_centered[i][j];

            // Update electron velocities (generalized Ohm's law) with Hall term
            electron_velocity_x[i][j] = -Ex_val / QE - grad_pressure_x / (QE * density) + (electron_velocity_y[i][j] * Bz_val) / MU_0;
            electron_velocity_y[i][j] = -Ey_val / QE - grad_pressure_y / (QE * density) - (electron_velocity_x[i][j] * Bz_val) / MU_0;
        }
    }
}

// Gathering Data
void gatherParticleDataMPI(const std::vector<std::vector<double>>& local_data,
                           std::vector<std::vector<double>>& global_data) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_count = local_data.size();
    std::vector<int> counts(size);

    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_data.clear(); // Clear previous content

        // Resize global_data based on the gathered counts
        global_data.reserve(std::accumulate(counts.begin(), counts.end(), 0));
    }

    // Send local data to rank 0
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            if (r != 0) {
                // Receive data from other ranks
                MPI_Status status;
                int count;
                MPI_Probe(r, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_DOUBLE, &count); // Get the number of doubles

                std::vector<std::vector<double>> received_data(counts[r], std::vector<double>(2)); // Assuming 2D data: x, y

                for (int i = 0; i < counts[r]; i++) {
                    MPI_Recv(received_data[i].data(), 2, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &status);
                }

                // Append received data to global data
                global_data.insert(global_data.end(), received_data.begin(), received_data.end());
            } else {
                // Append local data to global data for rank 0
                global_data.insert(global_data.end(), local_data.begin(), local_data.end());
            }
        }
    } else {
        // Send data from other ranks to rank 0
        for (const auto& particle : local_data) {
            MPI_Send(particle.data(), particle.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}
// Particle Injection in MPI
void injectParticles(int rank, int size, int px, int py, 
                     int local_NY, double DX, double DY, int GHOST_CELLS, double DT, 
                     double injection_area, int current_time_step,
                     std::vector<std::vector<double>>& particle_positions, 
                     std::vector<std::vector<double>>& particle_velocities) {
    // Determine the local domain boundaries in the y-direction
    int y_rank = rank / px;  // Rank's position in the y-direction
    //int local_NY = NY / py;  // Local grid size in the y-direction (Changed to argument)
    double start_y = y_rank * local_NY * DY;  // Starting physical coordinate (No change here)
    double end_y = start_y + (local_NY - 2 * GHOST_CELLS) * DY;  // Ending physical coordinate (local_NY instead of NY)

    // Compute injection parameters
    double injection_density = MIN_SOLAR_WIND_DENSITY +
                               (std::rand() / double(RAND_MAX)) * (MAX_SOLAR_WIND_DENSITY - MIN_SOLAR_WIND_DENSITY);
    double injection_speed = MIN_SOLAR_WIND_SPEED +
                             (std::rand() / double(RAND_MAX)) * (MAX_SOLAR_WIND_SPEED - MIN_SOLAR_WIND_SPEED);

    int num_injected_particles = static_cast<int>(injection_density * injection_area * DT / size);

    for (int i = 0; i < num_injected_particles; ++i) {
        // Inject particles at the inflow boundary (x = 0 for all processes)
        double x = 0.0;

        // Calculate y-coordinate uniformly across the valid range of the subdomain
        double y_min = start_y; // (Removed the GHOST_CELLS * DY addition from here)
        double y_max = end_y; // (Removed the GHOST_CELLS * DY subtraction from here)
        double y = y_min + (std::rand() / double(RAND_MAX)) * (y_max - y_min);

        // Add randomness to the velocity
        double vx = injection_speed + (std::rand() / double(RAND_MAX) - 0.5) * injection_speed * 0.05;
        double vy = (std::rand() / double(RAND_MAX) - 0.5) * injection_speed * 0.1;

        particle_positions.push_back({x, y});
        particle_velocities.push_back({vx, vy});
    }

    // Periodically gather particle data for monitoring (if applicable)
    if (rank == 0 && current_time_step % MONITOR_INTERVAL == 0) {
        std::vector<std::vector<double>> global_particle_positions;
        std::vector<std::vector<double>> global_particle_velocities;

        gatherParticleDataMPI(particle_positions, global_particle_positions);
        gatherParticleDataMPI(particle_velocities, global_particle_velocities);

        // Log particle data
        if (!global_particle_positions.empty()) {
            std::cout << "Time Step " << current_time_step
                      << ": Number of particles after injection: "
                      << global_particle_positions.size() << std::endl;
        }
    }
}

// Boris Push
void borisPushParticlesMPI() {
    std::vector<std::vector<double>> local_new_positions;
    std::vector<std::vector<double>> local_new_velocities;

    // Calculate start_x and start_y correctly for this rank
    int start_x = (rank % px) * (NX / px);
    int start_y = (rank / px) * (NY / py);

    // Update particle positions and velocities
    for (size_t p = 0; p < particle_positions.size(); ++p) {
        auto pos = particle_positions[p];
        auto vel = particle_velocities[p];

        // Update position
        pos[0] += vel[0] * DT;
        pos[1] += vel[1] * DT;

        // Assertions to check if positions are within bounds
        assert(pos[0] >= -1e-5 && pos[0] < NX * DX + 1e-5 && "Particle x-position out of bounds");
        assert(pos[1] >= -1e-5 && pos[1] < NY * DY + 1e-5 && "Particle y-position out of bounds");

        // Boundary conditions for x (horizontal) direction
        if (pos[0] < 0.0) {
            // Reflect at the left boundary
            pos[0] = 0.0;
            vel[0] = std::abs(vel[0]); // Reverse velocity direction
        } else if (pos[0] >= (start_x + local_NX) * DX) {
            // Remove particles leaving through the right boundary of this process's subdomain
            continue; // Skip storing this particle
        }

        // Boundary conditions for y (vertical) direction
        if (pos[1] < 0.0) {
            // Reflect at the bottom boundary
            pos[1] = 0.0;
            vel[1] = std::abs(vel[1]);
        } else if (pos[1] >= (start_y + local_NY) * DY) {
            // Remove particles leaving through the top boundary of this process's subdomain
            continue; // Skip storing this particle
        }

        // Store the updated position and velocity in local arrays
        local_new_positions.push_back(pos);
        local_new_velocities.push_back(vel);
    }

    // Update local particle positions and velocities
    particle_positions = local_new_positions;
    particle_velocities = local_new_velocities;

    // Gather particles leaving the subdomain
    std::vector<double> outgoing_buffer;
    for (size_t p = 0; p < particle_positions.size(); ++p) {
        if (particle_positions[p][0] >= (start_x + local_NX) * DX - 1e-5) {
            // Append position and velocity to the outgoing buffer
            outgoing_buffer.push_back(particle_positions[p][0]);
            outgoing_buffer.push_back(particle_positions[p][1]);
            outgoing_buffer.push_back(particle_velocities[p][0]);
            outgoing_buffer.push_back(particle_velocities[p][1]);

            // Remove this particle from local storage
            particle_positions.erase(particle_positions.begin() + p);
            particle_velocities.erase(particle_velocities.begin() + p);
            --p; // Adjust loop counter after removal
        }
    }

    // Exchange outgoing particles with neighboring processes
    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Send outgoing buffer size first
    int outgoing_size = outgoing_buffer.size();
    MPI_Send(&outgoing_size, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

    // Send outgoing particles as a single buffer
    if (outgoing_size > 0) {
        MPI_Send(outgoing_buffer.data(), outgoing_size, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD);
    }

    // Receive incoming particles from the previous rank
    int incoming_size = 0;
    MPI_Recv(&incoming_size, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (incoming_size > 0) {
        std::vector<double> incoming_buffer(incoming_size);
        MPI_Recv(incoming_buffer.data(), incoming_size, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Parse incoming buffer and add particles to local arrays
        for (int i = 0; i < incoming_size; i += 4) {
            particle_positions.push_back({incoming_buffer[i], incoming_buffer[i + 1]});
            particle_velocities.push_back({incoming_buffer[i + 2], incoming_buffer[i + 3]});
        }
    }

    // Debugging output
    std::cout << "Rank " << rank << ": Particle count after MPI exchange: " << particle_positions.size() << std::endl;
}

// Utility function definitions
void smoothDivergenceField(const std::vector<std::vector<double>>& divergence, std::vector<std::vector<double>>& smoothed_divergence) {
    int rows = divergence.size();
    int cols = divergence[0].size();

    // Ensure smoothed_divergence has the correct size
    smoothed_divergence.resize(rows, std::vector<double>(cols, 0.0));

    // Apply smoothing logic to the entire grid, excluding ghost cells
    for (int i = GHOST_CELLS; i < rows - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < cols - GHOST_CELLS; ++j) {
            // Average with neighboring cells
            smoothed_divergence[i][j] = 0.25 * (divergence[i + 1][j] + divergence[i - 1][j] +
                                                divergence[i][j + 1] + divergence[i][j - 1]);
        }
    }

    exchangeGhostCellsMPI(smoothed_divergence, "smoothed_divergence", local_NX, local_NY, rank, size);
}

// Ion Fluid Solver
void updateIonDensity() {
    if (ion_density_centered.size() != local_NX || 
        ion_density_centered[0].size() != local_NY + 2 * GHOST_CELLS) {
        std::cerr << "Error on rank " << rank << ": ion_density_centered dimension mismatch in updateIonDensity. "
                  << "Expected: " << local_NX << " x " << (local_NY + 2 * GHOST_CELLS)
                  << ", but got: " << ion_density_centered.size() << " x " 
                  << (ion_density_centered.empty() ? 0 : ion_density_centered[0].size()) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<std::vector<double>> new_density = ion_density_centered;

    exchangeGhostCellsMPI(ion_density_centered, "ion_density", local_NX, local_NY, rank, size);

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            double flux_x = (ion_density_centered[i][j] * ion_velocity_x_centered[i][j] - 
                             ion_density_centered[i - 1][j] * ion_velocity_x_centered[i - 1][j]) / DX;
            double flux_y = (ion_density_centered[i][j] * ion_velocity_y_centered[i][j] - 
                             ion_density_centered[i][j - 1] * ion_velocity_y_centered[i][j - 1]) / DY;

            new_density[i][j] -= DT * (flux_x + flux_y);
            new_density[i][j] = std::max(new_density[i][j], 1e-10);
        }
    }

    ion_density_centered = new_density;

    exchangeGhostCellsMPI(ion_density_centered, "ion_density", local_NX, local_NY, rank, size);
}

void updateIonVelocity() {
    std::vector<std::vector<double>> new_velocity_x = ion_velocity_x_centered; // Use _centered versions
    std::vector<std::vector<double>> new_velocity_y = ion_velocity_y_centered; // Use _centered versions

    // Exchange ghost cells for required fields, including the staggered Ex, Ey, Bx, By, Bz
    exchangeGhostCellsMPI(ion_pressure_centered, "ion_pressure", local_NX, local_NY, rank, size); // Use _centered
    exchangeGhostCellsMPI(ion_velocity_x_centered, "ion_velocity_x", local_NX, local_NY, rank, size); // Use _centered
    exchangeGhostCellsMPI(ion_velocity_y_centered, "ion_velocity_y", local_NX, local_NY, rank, size); // Use _centered
    exchangeGhostCellsMPI(Ex_staggered, "Ex", local_NX, local_NY, rank, size); // Use _staggered
    exchangeGhostCellsMPI(Ey_staggered, "Ey", local_NX, local_NY, rank, size); // Use _staggered
    exchangeGhostCellsMPI(Bz_centered, "Bz", local_NX, local_NY, rank, size); // Use _centered
    exchangeGhostCellsMPI(Bx_staggered, "Bx", local_NX, local_NY, rank, size); // Use _staggered
    exchangeGhostCellsMPI(By_staggered, "By", local_NX, local_NY, rank, size); // Use _staggered

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) { // Use local_NY
            // Ensure numerical stability by enforcing a minimum density threshold
            double local_density = std::max(ion_density_centered[i][j], 1e-10); // Use _centered

            // Momentum equation: ∂v/∂t + v·∇v = -(1/m_i)∇p + (q_i/m_i)(E + v × B)
            double pressure_grad_x = (ion_pressure_centered[i + 1][j] - ion_pressure_centered[i - 1][j]) / (2.0 * DX);
            double pressure_grad_y = (ion_pressure_centered[i][j + 1] - ion_pressure_centered[i][j - 1]) / (2.0 * DY);

            // Interpolate Ex and Ey to cell centers (i, j)
            double Ex_center = (i < Ex_staggered.size() - 1) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i + 1][j]) : Ex_staggered[i][j];
            double Ey_center = (j < Ey_staggered[0].size() - 1) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j + 1]) : Ey_staggered[i][j];

            // Interpolate Bx and By to cell centers
            double Bx_center = (i > 0 && i < Bx_staggered.size()) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i-1][j]) : Bx_staggered[i][j];
            double By_center = (j > 0 && j < By_staggered[0].size()) ? 0.5 * (By_staggered[i][j] + By_staggered[i][j-1]) : By_staggered[i][j];

            // Use Bz_centered directly since it's already at the cell center
            double Bz_val = Bz_centered[i][j];

            // Calculate Lorentz force with interpolated values
            double lorentz_force_x = QI / MI * (Ex_center + ion_velocity_y_centered[i][j] * Bz_val);
            double lorentz_force_y = QI / MI * (Ey_center - ion_velocity_x_centered[i][j] * Bz_val);

            // Calculate advective terms using centered velocities
            double advective_x = ion_velocity_x_centered[i][j] * (ion_velocity_x_centered[i + 1][j] - ion_velocity_x_centered[i - 1][j]) / (2.0 * DX);
            double advective_y = ion_velocity_y_centered[i][j] * (ion_velocity_y_centered[i][j + 1] - ion_velocity_y_centered[i][j - 1]) / (2.0 * DY);

            // Update velocities
            new_velocity_x[i][j] -= DT * (advective_x + pressure_grad_x / local_density - lorentz_force_x);
            new_velocity_y[i][j] -= DT * (advective_y + pressure_grad_y / local_density - lorentz_force_y);
        }
    }

    // Update velocities locally
    ion_velocity_x_centered = new_velocity_x;
    ion_velocity_y_centered = new_velocity_y;

    // Exchange ghost cells to maintain consistency across subdomains
    exchangeGhostCellsMPI(ion_velocity_x_centered, "ion_velocity_x", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_y_centered, "ion_velocity_y", local_NX, local_NY, rank, size);
}

void updateIonPressureMPI() {
    std::vector<std::vector<double>> new_pressure = ion_pressure_centered; // Use the _centered version

    // Exchange ghost cells for the centered ion pressure and velocities
    exchangeGhostCellsMPI(ion_pressure_centered, "ion_pressure", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_x_centered, "ion_velocity_x", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_y_centered, "ion_velocity_y", local_NX, local_NY, rank, size);

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Use ion_pressure_centered for consistency
            double local_pressure = std::max(ion_pressure_centered[i][j], 1e-10);

            // Calculate fluxes using centered ion velocities and pressure
            double flux_x = (ion_velocity_x_centered[i][j] * local_pressure - ion_velocity_x_centered[i - 1][j] * ion_pressure_centered[i - 1][j]) / DX;
            double flux_y = (ion_velocity_y_centered[i][j] * local_pressure - ion_velocity_y_centered[i][j - 1] * ion_pressure_centered[i][j - 1]) / DY;

            // Calculate velocity divergence using centered ion velocities
            double velocity_div = (ion_velocity_x_centered[i + 1][j] - ion_velocity_x_centered[i - 1][j]) / (2.0 * DX)
                                + (ion_velocity_y_centered[i][j + 1] - ion_velocity_y_centered[i][j - 1]) / (2.0 * DY);

            double smoothed_velocity_div = smoothDivergence(velocity_div);

            // Update pressure, ensuring it remains non-negative
            new_pressure[i][j] -= DT * (flux_x + flux_y + GAMMA * local_pressure * smoothed_velocity_div);
            new_pressure[i][j] = std::max(new_pressure[i][j], 1e-10);
        }
    }

    // Update the ion pressure with the new values
    ion_pressure_centered = new_pressure;

    // Re-exchange ghost cells for ion_pressure_centered after updating
    exchangeGhostCellsMPI(ion_pressure_centered, "ion_pressure", local_NX, local_NY, rank, size);

    // Diagnostics to track the maximum and minimum pressure across all processes
    double local_max_pressure = 0.0;
    double local_min_pressure = std::numeric_limits<double>::max();

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            local_max_pressure = std::max(local_max_pressure, ion_pressure_centered[i][j]);
            local_min_pressure = std::min(local_min_pressure, ion_pressure_centered[i][j]);
        }
    }

    double global_max_pressure, global_min_pressure;
    MPI_Allreduce(&local_max_pressure, &global_max_pressure, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min_pressure, &global_min_pressure, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Pressure Diagnostics: Max = " << global_max_pressure
                  << ", Min = " << global_min_pressure << std::endl;
    }
}

void updateIonFluid() {
    // Temporary storage for updated fields
    std::vector<std::vector<double>> new_density = ion_density_centered;
    std::vector<std::vector<double>> new_velocity_x = ion_velocity_x_centered;
    std::vector<std::vector<double>> new_velocity_y = ion_velocity_y_centered;
    std::vector<std::vector<double>> new_pressure = ion_pressure_centered;

    // Exchange ghost cells for Bz_centered before computing derivatives
    exchangeGhostCellsMPI(Bz_centered, "Bz", local_NX, local_NY, rank, size);

    // Compute derivatives for advanced terms (dBz_dx, dBz_dy) locally using MPI variant
    std::vector<std::vector<double>> dBz_dx(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> dBz_dy(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    computeDerivativesMPI(Bz_centered, dBz_dx, dBz_dy, local_NX, local_NY, rank, size); // Use Bz_centered

    // Exchange ghost cells for Ex, Ey, and other fields used in the update
    exchangeGhostCellsMPI(Ex_staggered, "Ex", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Ey_staggered, "Ey", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_density_centered, "ion_density", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_x_centered, "ion_velocity_x", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_y_centered, "ion_velocity_y", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_pressure_centered, "ion_pressure", local_NX, local_NY, rank, size);

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) { // Use local_NY here
            // Ensure stability by limiting ion_density
            double local_density = std::max(ion_density_centered[i][j], 1e-10);

            // Continuity equation: ∂n/∂t + ∇·(n*v) = 0
            double flux_x = (ion_density_centered[i][j] * ion_velocity_x_centered[i][j] - ion_density_centered[i - 1][j] * ion_velocity_x_centered[i - 1][j]) / DX;
            double flux_y = (ion_density_centered[i][j] * ion_velocity_y_centered[i][j] - ion_density_centered[i][j - 1] * ion_velocity_y_centered[i][j - 1]) / DY;
            new_density[i][j] -= DT * (flux_x + flux_y);

            // Momentum equation: ∂v/∂t + v·∇v = -(1/ρ)∇p + (q/m)(E + v × B)
            double pressure_grad_x = (ion_pressure_centered[i + 1][j] - ion_pressure_centered[i - 1][j]) / (2.0 * DX);
            double pressure_grad_y = (ion_pressure_centered[i][j + 1] - ion_pressure_centered[i][j - 1]) / (2.0 * DY);

            // Interpolate Ex and Ey to cell centers (i, j)
            double Ex_center = (i < Ex_staggered.size() - 1) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i + 1][j]) : Ex_staggered[i][j];
            double Ey_center = (j < Ey_staggered[0].size() - 1) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j + 1]) : Ey_staggered[i][j];

            // Interpolate Bx and By to cell centers for Lorentz force calculation
            double Bx_center = (i > 0 && i < Bx_staggered.size()) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i - 1][j]) : Bx_staggered[i][j];
            double By_center = (j > 0 && j < By_staggered[0].size()) ? 0.5 * (By_staggered[i][j] + By_staggered[i][j - 1]) : By_staggered[i][j];

            double lorentz_force_x = QI / MI * (Ex_center + ion_velocity_y_centered[i][j] * Bz_centered[i][j]);
            double lorentz_force_y = QI / MI * (Ey_center - ion_velocity_x_centered[i][j] * Bz_centered[i][j]);

            double advective_x = ion_velocity_x_centered[i][j] * (ion_velocity_x_centered[i + 1][j] - ion_velocity_x_centered[i - 1][j]) / (2.0 * DX);
            double advective_y = ion_velocity_y_centered[i][j] * (ion_velocity_y_centered[i][j + 1] - ion_velocity_y_centered[i][j - 1]) / (2.0 * DY);

            new_velocity_x[i][j] -= DT * (advective_x + pressure_grad_x / local_density - lorentz_force_x);
            new_velocity_y[i][j] -= DT * (advective_y + pressure_grad_y / local_density - lorentz_force_y);

            // Energy equation: ∂p/∂t + v·∇p + γp∇·v = q * n (v_e - v_i) · E (heat generation)
            double flux_p_x = (ion_velocity_x_centered[i][j] * ion_pressure_centered[i][j] - ion_velocity_x_centered[i - 1][j] * ion_pressure_centered[i - 1][j]) / DX;
            double flux_p_y = (ion_velocity_y_centered[i][j] * ion_pressure_centered[i][j] - ion_velocity_y_centered[i][j - 1] * ion_pressure_centered[i][j - 1]) / DY;

            double velocity_div = (ion_velocity_x_centered[i + 1][j] - ion_velocity_x_centered[i - 1][j]) / (2.0 * DX)
                                + (ion_velocity_y_centered[i][j + 1] - ion_velocity_y_centered[i][j - 1]) / (2.0 * DY);

            double heat_generation = QI * local_density * (electron_velocity_x[i][j] - ion_velocity_x_centered[i][j]) * Ex_center
                                   + QI * local_density * (electron_velocity_y[i][j] - ion_velocity_y_centered[i][j]) * Ey_center;

            new_pressure[i][j] -= DT * (flux_p_x + flux_p_y + GAMMA * ion_pressure_centered[i][j] * velocity_div - heat_generation / local_density);
        }
    }

    // Update fields locally
    ion_density_centered = new_density;
    ion_velocity_x_centered = new_velocity_x;
    ion_velocity_y_centered = new_velocity_y;
    ion_pressure_centered = new_pressure;

    // Exchange ghost cells to maintain consistency across processes
    exchangeGhostCellsMPI(ion_density_centered, "ion_density", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_x_centered, "ion_velocity_x", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_velocity_y_centered, "ion_velocity_y", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(ion_pressure_centered, "ion_pressure", local_NX, local_NY, rank, size);
}
// Reconnection
double calculateReconnectionRateMPI(
    const std::vector<std::vector<double>>& Ex_staggered,
    const std::vector<std::vector<double>>& Ey_staggered,
    const std::vector<std::vector<double>>& Bx_staggered,
    const std::vector<std::vector<double>>& By_staggered,
    const std::vector<std::vector<double>>& Bz_centered,
    int global_x_point_i, int global_x_point_j) {

    double local_E_rec = 0.0;

    // Convert global indices to local indices for this rank
    int local_x_point_i = global_x_point_i - (rank % px) * (NX / px) + GHOST_CELLS;
    int local_x_point_j = global_x_point_j - (rank / px) * (NY / py) + GHOST_CELLS;

    // Check if the X-point lies within the local subdomain
    if (local_x_point_i >= GHOST_CELLS && local_x_point_i < local_NX - GHOST_CELLS &&
        local_x_point_j >= GHOST_CELLS && local_x_point_j < local_NY + GHOST_CELLS) {

        // Interpolate Bx to the cell center (i, j)
        double Bx_xp = 0.5 * (Bx_staggered[local_x_point_i][local_x_point_j] + Bx_staggered[local_x_point_i - 1][local_x_point_j]);
        // Interpolate By to the cell center (i, j)
        double By_xp = 0.5 * (By_staggered[local_x_point_i][local_x_point_j] + By_staggered[local_x_point_i][local_x_point_j - 1]);
        // Bz is already at the cell center
        double Bz_xp = Bz_centered[local_x_point_i][local_x_point_j];

        // Access Ex_staggered and Ey_staggered at the correct staggered locations (i+1/2, j) and (i, j+1/2) respectively
        // Assuming Ex_staggered at (i+1/2, j) and Ey_staggered at (i, j+1/2)
        double Ex_xp = Ex_staggered[local_x_point_i][local_x_point_j]; 
        double Ey_xp = Ey_staggered[local_x_point_i][local_x_point_j];

        // Magnetic field magnitude at X-point
        double B_magnitude = std::sqrt(Bx_xp * Bx_xp + By_xp * By_xp + Bz_xp * Bz_xp);

        if (B_magnitude > 1e-10) {
            // Reconnection rate (Ez at the X-point)
            local_E_rec = std::abs(Ey_xp); 
        } else {
            local_E_rec = 0.0; // Avoid division by zero
        }
    }

    // Reduce to find the global reconnection rate (summing contributions)
    double global_E_rec = 0.0;
    MPI_Reduce(&local_E_rec, &global_E_rec, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_E_rec;
}

// Calculate Poynting Flux
double calculatePoyntingFluxMPI(
    const std::vector<std::vector<double>>& Ex_staggered,
    const std::vector<std::vector<double>>& Ey_staggered,
    const std::vector<std::vector<double>>& Bz_centered
) {
    double local_poynting_flux = 0.0;

    // Calculate Poynting flux for the local subdomain
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Interpolate Ex and Ey to cell centers (i, j) for calculation
            double Ex_center = (i < Ex_staggered.size() - 1) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i + 1][j]) : Ex_staggered[i][j];
            double Ey_center = (j < Ey_staggered[0].size() - 1) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j + 1]) : Ey_staggered[i][j];

            // Bz is already at cell centers, so no interpolation needed
            double Bz_val = Bz_centered[i][j];

            // Cross product of E and B at cell centers
            double Sx = Ey_center * Bz_val / MU_0;
            double Sy = -Ex_center * Bz_val / MU_0;

            // Calculate magnitude of Poynting flux at cell center
            double S_magnitude = std::sqrt(Sx * Sx + Sy * Sy);

            // Accumulate Poynting flux over the local domain
            local_poynting_flux += S_magnitude * DX * DY;
        }
    }

    // Reduce local Poynting flux to global sum
    double global_poynting_flux = 0.0;
    MPI_Reduce(&local_poynting_flux, &global_poynting_flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Log diagnostics on rank 0 for validation
    if (rank == 0) {
        std::cout << "Global Poynting Flux: " << global_poynting_flux << std::endl;
    }

    return global_poynting_flux;
}

// Adaptive Time Step
double computeAdaptiveTimestepMPI() {
    const double courant_number = 0.5; // CFL condition
    const double min_density_threshold = 1e-10; // Avoid division by zero
    const double max_velocity_threshold = 1e6;  // Prevent overly restrictive time steps
    double local_max_velocity = 0.0;

    // Compute the maximum velocity in the local domain
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Ensure ion density is above the minimum threshold
            double local_density = std::max(ion_density_centered[i][j], min_density_threshold);

            // Calculate velocity magnitude (using cell-centered velocities)
            double velocity_magnitude = std::sqrt(
                ion_velocity_x_centered[i][j] * ion_velocity_x_centered[i][j] +
                ion_velocity_y_centered[i][j] * ion_velocity_y_centered[i][j]
            );

            // Interpolate Bx and By to cell centers for Alfven speed calculation
            double Bx_center = (i > 0 && i < Bx_staggered.size()) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i - 1][j]) : Bx_staggered[i][j];
            double By_center = (j > 0 && j < By_staggered[0].size()) ? 0.5 * (By_staggered[i][j] + By_staggered[i][j - 1]) : By_staggered[i][j];

            // Calculate Alfven speed (using cell-centered Bz)
            double alfven_speed = std::sqrt(
                (Bx_center * Bx_center + By_center * By_center + Bz_centered[i][j] * Bz_centered[i][j]) / (MU_0 * local_density)
            );

            // Select the maximum between velocity magnitude and Alfven speed
            double local_max = std::max(velocity_magnitude, alfven_speed);

            // Safeguard against excessively high velocities
            if (local_max > max_velocity_threshold) {
                local_max = max_velocity_threshold;
            }

            local_max_velocity = std::max(local_max_velocity, local_max);
        }
    }

    // Reduce to find the global maximum velocity across all processes
    double global_max_velocity = 0.0;
    MPI_Allreduce(&local_max_velocity, &global_max_velocity, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Avoid division by zero and return the adaptive timestep
    if (global_max_velocity < 1e-10) {
        global_max_velocity = 1e-10; // Minimum velocity threshold
    }

    return courant_number * std::min(DX, DY) / global_max_velocity;
}

// Energy Change
double calculateEnergyChangeMPI() {
    double local_delta_w = 0.0;

    // Exchange ghost cells for Jx_staggered, Jy_staggered, Ex_staggered, and Ey_staggered
    exchangeGhostCellsMPI(Jx_staggered, "Jx_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Jy_staggered, "Jy_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Ex_staggered, "Ex_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Ey_staggered, "Ey_staggered", local_NX, local_NY, rank, size);

    // Compute the energy change in the local domain
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Interpolate Jx and Ex to the cell centers (i, j)
            double Jx_interp = (i > 0 && i < Jx_staggered.size()) ? 0.5 * (Jx_staggered[i][j] + Jx_staggered[i-1][j]) : 0.0;
            double Ex_interp = (i > 0 && i < Ex_staggered.size()) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i-1][j]) : Ex_staggered[i][j];

            // Interpolate Jy and Ey to the cell centers (i, j)
            double Jy_interp = (j > 0 && j < Jy_staggered[0].size()) ? 0.5 * (Jy_staggered[i][j] + Jy_staggered[i][j-1]) : 0.0;
            double Ey_interp = (j > 0 && j < Ey_staggered[0].size()) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j-1]) : Ey_staggered[i][j];

            // Calculate j_dot_e at the cell center
            double j_dot_e = Jx_interp * Ex_interp + Jy_interp * Ey_interp;
            local_delta_w += j_dot_e * DX * DY;
        }
    }

    // Reduce to find the global energy change across all processes
    double global_delta_w = 0.0;
    MPI_Reduce(&local_delta_w, &global_delta_w, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_delta_w;
}

void calculateDiagnosticsMPI(double current_time) {
    // Step 1: Compute magnetic divergence locally
    std::vector<std::vector<double>> div_B(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    computeMagneticDivergenceMPI(Bx_staggered, By_staggered, Bz_centered, div_B, local_NX, local_NY, rank, size);

    // Step 2: Solve magnetic scalar potential locally using multigrid
    auto local_phi_B = computeMagneticScalarPotentialMPI(div_B, local_NX, local_NY, multigrid_levels, rank, size);

    // Step 3: Identify X-point locally
    std::pair<int, int> local_x_point = findXPointFromPotentialMPI(local_phi_B);
    int local_x_point_i = local_x_point.first;
    int local_x_point_j = local_x_point.second;

    // Step 4: Reduce X-point coordinates globally
    int global_x_point_i = -1, global_x_point_j = -1;
    MPI_Allreduce(&local_x_point_i, &global_x_point_i, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_x_point_j, &global_x_point_j, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Time Step Diagnostics: " << current_time << std::endl;
    }

    if (global_x_point_i != -1 && global_x_point_j != -1) {
        // Calculate reconnection rate and inflow velocity locally
        double local_reconnection_rate = calculateReconnectionRateMPI(Ex_staggered, Ey_staggered, Bx_staggered, By_staggered, Bz_centered, global_x_point_i, global_x_point_j);
        double local_inflow_velocity = calculateInflowVelocityMPI(ion_velocity_x_centered, ion_velocity_y_centered, global_x_point_i, global_x_point_j, local_NX, local_NY, rank, size);

        // Reduce reconnection rate and inflow velocity globally
        double global_reconnection_rate = 0.0, global_inflow_velocity = 0.0;
        MPI_Reduce(&local_reconnection_rate, &global_reconnection_rate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_inflow_velocity, &global_inflow_velocity, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "X-Point Coordinates: (" << global_x_point_i << ", " << global_x_point_j << ")" << std::endl;
            std::cout << "Reconnection Rate (Ez at X-point): " << global_reconnection_rate << std::endl;
            std::cout << "Inflow Velocity: " << global_inflow_velocity << std::endl;
        }
    } else if (rank == 0) {
        std::cout << "X-point not found in this timestep!" << std::endl;
    }

    // Step 5: Compute energy diagnostics locally
    double local_electric_energy = 0.0, local_magnetic_energy = 0.0, local_thermal_energy = 0.0, local_kinetic_energy = 0.0;
    double local_max_pressure = 0.0, local_min_pressure = std::numeric_limits<double>::max();

    // Use the correct staggered variables and local_NY for energy calculations
    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS; ++i) {
        for (int j = GHOST_CELLS; j < local_NY + GHOST_CELLS; ++j) {
            // Interpolate Ex and Ey to cell centers for energy calculation
            double Ex_center = (i < Ex_staggered.size() - 1) ? 0.5 * (Ex_staggered[i][j] + Ex_staggered[i + 1][j]) : Ex_staggered[i][j];
            double Ey_center = (j < Ey_staggered[0].size() - 1) ? 0.5 * (Ey_staggered[i][j] + Ey_staggered[i][j + 1]) : Ey_staggered[i][j];

            // Interpolate Bx and By to cell centers for energy calculation
            double Bx_center = (i > 0 && i < Bx_staggered.size()) ? 0.5 * (Bx_staggered[i][j] + Bx_staggered[i - 1][j]) : Bx_staggered[i][j];
            double By_center = (j > 0 && j < By_staggered[0].size()) ? 0.5 * (By_staggered[i][j] + By_staggered[i][j - 1]) : By_staggered[i][j];

            local_electric_energy += 0.5 * EPS_0 * (Ex_center * Ex_center + Ey_center * Ey_center) * DX * DY;
            local_magnetic_energy += 0.5 * (Bx_center * Bx_center + By_center * By_center + Bz_centered[i][j] * Bz_centered[i][j]) / MU_0 * DX * DY;
            local_thermal_energy += ion_pressure_centered[i][j] / (GAMMA - 1) * DX * DY;

            // Track pressure diagnostics
            local_max_pressure = std::max(local_max_pressure, ion_pressure_centered[i][j]);
            local_min_pressure = std::min(local_min_pressure, ion_pressure_centered[i][j]);
        }
    }

    // Compute kinetic energy locally
    for (size_t p = 0; p < particle_positions.size(); ++p) {
        double vx = particle_velocities[p][0];
        double vy = particle_velocities[p][1];
        local_kinetic_energy += 0.5 * ME * (vx * vx + vy * vy);
    }

    // Step 6: Reduce energy diagnostics globally
    double global_electric_energy = 0.0, global_magnetic_energy = 0.0, global_thermal_energy = 0.0, global_kinetic_energy = 0.0;
    double global_max_pressure = 0.0, global_min_pressure = 0.0;
    MPI_Reduce(&local_electric_energy, &global_electric_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_magnetic_energy, &global_magnetic_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_thermal_energy, &global_thermal_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_pressure, &global_max_pressure, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min_pressure, &global_min_pressure, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    // Step 7: Calculate Poynting flux locally using the corrected function
    double local_poynting_flux = calculatePoyntingFluxMPI(Ex_staggered, Ey_staggered, Bz_centered);

    // Reduce Poynting flux globally
    double global_poynting_flux = 0.0;
    MPI_Reduce(&local_poynting_flux, &global_poynting_flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Step 8: Output diagnostics on rank 0
    if (rank == 0) {
        std::cout << "Electric Energy: " << global_electric_energy << std::endl;
        std::cout << "Magnetic Energy: " << global_magnetic_energy << std::endl;
        std::cout << "Kinetic Energy: " << global_kinetic_energy << std::endl;
        std::cout << "Thermal Energy: " << global_thermal_energy << std::endl;
        std::cout << "Poynting Flux: " << global_poynting_flux << std::endl;
        std::cout << "Max Pressure: " << global_max_pressure << std::endl;
        std::cout << "Min Pressure: " << global_min_pressure << std::endl;

        // Store energy values for trend analysis
        electric_energy_history.push_back(global_electric_energy);
        magnetic_energy_history.push_back(global_magnetic_energy);
        kinetic_energy_history.push_back(global_kinetic_energy);
        thermal_energy_history.push_back(global_thermal_energy);
        total_energy_history.push_back(global_electric_energy + global_magnetic_energy + global_kinetic_energy + global_thermal_energy);
    }
}

// Calculate current density
void calculateParticleCurrentMPI(std::vector<std::vector<double>>& Jx_staggered,
                                  std::vector<std::vector<double>>& Jy_staggered) {
    // Initialize local current densities to zero on the staggered grid
    Jx_staggered.assign(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0)); // Jx at x-faces
    Jy_staggered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0)); // Jy at y-faces

    // Get the MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate start_x and start_y for each rank
    int start_x = (rank % px) * (NX / px);
    int start_y = (rank / px) * (NY / py);

    // Loop over local particles and update local current density
    for (size_t p = 0; p < particle_positions.size(); ++p) {
        double x = particle_positions[p][0];
        double y = particle_positions[p][1];
        double vx = particle_velocities[p][0];
        double vy = particle_velocities[p][1];

        // Map particle position to the staggered local grid
        int i = static_cast<int>((x / DX) - start_x + GHOST_CELLS); // Local index for x-faces
        int j = static_cast<int>((y / DY) - start_y + GHOST_CELLS); // Local index for y-nodes

        // Ensure particles are within the valid subdomain, considering staggered grid
        if (i >= 0 && i < local_NX + 1 && j >= 0 && j < local_NY + 2 * GHOST_CELLS) {
            Jx_staggered[i][j] += QE * vx; // Jx at x-face
        }

        // Adjust indices for Jy which is on y-faces
        int i_Jy = i; // Same as for Jx
        int j_Jy = j;   // Same as for Jx

        if (i_Jy >= 0 && i_Jy < local_NX && j_Jy >= 0 && j_Jy < local_NY + 1) {
            Jy_staggered[i_Jy][j_Jy] += QE * vy; // Jy at y-face
        }
    }

    // Normalize by grid cell volume
    double normalization_factor = DX * DY;
    for (int i = 0; i < local_NX + 1; ++i) {
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
            Jx_staggered[i][j] /= normalization_factor;
        }
    }

    for (int i = 0; i < local_NX; ++i) {
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS + 1; ++j) {
            Jy_staggered[i][j] /= normalization_factor;
        }
    }
}

void calculateFluidCurrent(std::vector<std::vector<double>>& Jx_staggered,
                           std::vector<std::vector<double>>& Jy_staggered) {
    // Reset current densities to zero, ensuring correct dimensions for staggered grid
    Jx_staggered.assign(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0)); // Jx at x-faces
    Jy_staggered.assign(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0)); // Jy at y-faces

    for (int i = GHOST_CELLS; i < local_NX - GHOST_CELLS + 1; ++i) {
        for (int j = GHOST_CELLS; j < local_NY - GHOST_CELLS + 1; ++j) {
            // Skip computation if ion_density_centered is too small (or below threshold for stability)
            if (ion_density_centered[i][j] <= 1e-10) {
                continue;
            }

            // Compute current densities at correct staggered locations
            // Jx at (i+1/2, j) - halfway between i and i+1
            if (i < local_NX - GHOST_CELLS) {
                Jx_staggered[i][j] = QI * ion_density_centered[i][j] * ion_velocity_x_centered[i][j];
            }

            // Jy at (i, j+1/2) - halfway between j and j+1
            if (j < local_NY - GHOST_CELLS) {
                Jy_staggered[i][j] = QI * ion_density_centered[i][j] * ion_velocity_y_centered[i][j];
            }
        }
    }
}

// Combine particle and fluid contributions
void calculateTotalCurrentMPI(std::vector<std::vector<double>>& Jx_staggered,
                               std::vector<std::vector<double>>& Jy_staggered) {
    // Particle current (calculated on the staggered grid)
    std::vector<std::vector<double>> Jx_particle(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> Jy_particle(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0));
    calculateParticleCurrentMPI(Jx_particle, Jy_particle);

    // Fluid current (calculated on the staggered grid)
    std::vector<std::vector<double>> Jx_fluid(local_NX + 1, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> Jy_fluid(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS + 1, 0.0));
    calculateFluidCurrent(Jx_fluid, Jy_fluid);

    // Combine currents locally, taking into account the staggered grid
    for (int i = 0; i < local_NX + 1; ++i) {
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
            if (i < Jx_staggered.size() && j < Jx_staggered[i].size()) {
                Jx_staggered[i][j] = Jx_particle[i][j] + Jx_fluid[i][j];
            }
        }
    }

    for (int i = 0; i < local_NX; ++i) {
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS + 1; ++j) {
            if (i < Jy_staggered.size() && j < Jy_staggered[i].size()) {
                Jy_staggered[i][j] = Jy_particle[i][j] + Jy_fluid[i][j];
            }
        }
    }
    
    // Exchange ghost cells to ensure proper updating at subdomain boundaries
    exchangeGhostCellsMPI(Jx_staggered, "Jx_staggered", local_NX, local_NY, rank, size);
    exchangeGhostCellsMPI(Jy_staggered, "Jy_staggered", local_NX, local_NY, rank, size);
}

// Time Loop
void timeLoop(const std::vector<WENOWeights>& weights,
              int rank, int size, int px, int py,
              std::vector<std::vector<double>>& vx,
              std::vector<std::vector<double>>& vy) {
    double current_time = 0.0;
    int current_time_step = 0; // Initialize time step counter

    // Variables to gather global particle data
    std::vector<std::vector<double>> global_particle_positions;
    std::vector<std::vector<double>> global_particle_velocities;

    // Batch of fields to synchronize ghost cells.
    // Includes all staggered field variables and centered variables
    std::vector<std::vector<std::vector<double>>> fields_to_sync_batch = {
        Ex_staggered, Ey_staggered, Bz_centered, Bx_staggered, By_staggered,
        Jx_staggered, Jy_staggered, Jz_centered,
        ion_density_centered, ion_velocity_x_centered, ion_velocity_y_centered,
        ion_pressure_centered, electron_velocity_x, electron_velocity_y, electron_pressure_centered
    };
    std::vector<std::string> field_types = {
        "Ex_staggered", "Ey_staggered", "Bz_centered", "Bx_staggered", "By_staggered",
        "Jx_staggered", "Jy_staggered", "Jz_centered",
        "ion_density_centered", "ion_velocity_x_centered", "ion_velocity_y_centered",
        "ion_pressure_centered", "electron_velocity_x", "electron_velocity_y", "electron_pressure_centered"
    };

    while (current_time < TOTAL_SIMULATION_TIME) {
        // Exchange ghost cells for all fields at once before any computations that depend on them
        exchangeGhostCellsBatchMPI(fields_to_sync_batch, field_types, local_NX, local_NY, rank, size);

        // Independent computations (do not depend on updated ghost cell values)
        // Step 2: Compute adaptive time step
        double local_dt = computeAdaptiveTimestepMPI();
        double dt;
        MPI_Allreduce(&local_dt, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // Step 3: Inject particles (use local_NY)
        injectParticles(rank, size, px, py, local_NY, DX, DY, GHOST_CELLS,
                        dt, injection_area, current_time_step, particle_positions, particle_velocities);

        // Gather particle data (if needed for diagnostics or output)
        if (rank == 0 && current_time_step % MONITOR_INTERVAL == 0){
            gatherParticleDataMPI(particle_positions, global_particle_positions);
            gatherParticleDataMPI(particle_velocities, global_particle_velocities);
            std::cout << "After Particle Injection: Total number of particles: "
                      << global_particle_positions.size() << std::endl;
        }

        // Step 4: Update ion properties (these functions should be using local_NY internally)
        updateIonDensity();
        updateIonVelocity();
        updateIonPressureMPI();
        updateIonFluid();

        // Dependent computations (after ghost cells are updated)
        // Step 5: Update electron velocity (use Ex_staggered, Ey_staggered, etc.)
        updateElectronVelocity(electron_velocity_x, electron_velocity_y, Ex_staggered, Ey_staggered, Bz_centered, electron_pressure_centered, ion_density_centered, local_NX, local_NY, rank, size);

        // Step 6: Calculate currents (use Jx_staggered and Jy_staggered)
        calculateParticleCurrentMPI(Jx_staggered, Jy_staggered);
        calculateFluidCurrent(Jx_staggered, Jy_staggered);
        calculateTotalCurrentMPI(Jx_staggered, Jy_staggered);

        // Step 7: Update fields
        updateFields(Ex_staggered, Ey_staggered, Bz_centered, Jx_staggered, Jy_staggered, Bx_staggered, By_staggered, vx, vy, electron_pressure_centered, local_NX, local_NY, rank, size);
        updateElectricFieldCT(Ex_staggered, Ey_staggered, Bx_staggered, By_staggered, Bz_centered, vx, vy, electron_pressure_centered, ion_density_centered, Jx_staggered, Jy_staggered, local_NX, local_NY);

        // Step 8: Perform divergence cleaning for E and B fields
        // Step 8.1: Compute divergence of E
        auto divergence_E = computeDivergenceMPI(Ex_staggered, Ey_staggered);

        // Step 8.2: Solve for potential phi_E using the multigrid solver
        auto phi_E = solvePoissonMultigridMPI(divergence_E, local_NX, local_NY, multigrid_levels, "phi_E" , rank, size);

        // Step 8.3: Compute divergence of B
        std::vector<std::vector<double>> div_B(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
        computeMagneticDivergenceMPI(Bx_staggered, By_staggered, Bz_centered, div_B, local_NX, local_NY, rank, size);

        // Step 8.4: Solve for potential phi_B using the multigrid solver
        auto phi_B = solvePoissonMultigridMPI(div_B, local_NX, local_NY, multigrid_levels, "phi_B", rank, size);

        // Step 8.5: Apply correction to B fields
        applyBorisCorrectionMPI(Bx_staggered, By_staggered, phi_B, local_NX, local_NY, rank, size);

        // Add divergence check for B fields at regular intervals
        if (current_time_step % 100 == 0) {
            double magnetic_divergence = computeMagneticDivergenceMPI(Bx_staggered, By_staggered, Bz_centered, div_B, local_NX, local_NY, rank, size);
            if (rank == 0) {
                std::cout << "Time Step " << current_time_step 
                          << " | Magnetic Divergence (Max): " << magnetic_divergence << std::endl;
            }
        }

        // Step 9: Apply WENO reconstruction
        applyWENO(Ex_staggered, weights, "Ex_staggered");
        applyWENO(Ey_staggered, weights, "Ey_staggered");
        applyWENO(Bz_centered, weights, "Bz_centered");

        // Step 10: Push particles
        borisPushParticlesMPI();

        // Step 11: Diagnostics
        if (current_time_step % DIAGNOSTIC_INTERVAL == 0) {
            auto x_point = findXPointFromPotentialMPI(phi_B);
            calculateDiagnosticsMPI(current_time);
            double reconnection_rate = calculateReconnectionRateMPI(Ex_staggered, Ey_staggered, Bx_staggered, By_staggered, Bz_centered, x_point.first, x_point.second);
            if (rank == 0) {
                std::cout << "Reconnection Rate: " << reconnection_rate << std::endl;
            }
        }

        // Step 12: Compute Poynting flux
        double poynting_flux = calculatePoyntingFluxMPI(Ex_staggered, Ey_staggered, Bz_centered);
        if (rank == 0) {
            std::cout << "Poynting Flux: " << poynting_flux << std::endl;
        }

        // Step 13: Calculate energy changes
        double local_energy_change = calculateEnergyChangeMPI();
        double global_energy_change;
        MPI_Allreduce(&local_energy_change, &global_energy_change, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Energy Change (ΔW): " << global_energy_change << std::endl;
        }

        // Step 14: Update simulation time
        current_time += dt;
        current_time_step++;
    }

    // Finalize simulation diagnostics
    if (rank == 0) {
        std::cout << "Simulation completed. Total time: " << current_time << " seconds." << std::endl;
    }
}

// Main
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize MPI-related structures and subdomain sizes
    initializeMPI();

    // Debug: Verify grid dimensions
    if (rank == 0) {
        std::cout << "Grid dimensions: NX = " << NX << ", NY = " << NY
                  << ", px = " << px << ", py = " << py << std::endl;
    }

    // Step 1: Staggered Grid Initialization
    initializeCTGridStructure(local_NX, local_NY);

    // Debug: Print array sizes after initialization
    std::cout << "Rank " << rank << ": Post-initializeCTGridStructure, ion_density_centered sizes = "
              << ion_density_centered.size() << " x "
              << (ion_density_centered.empty() ? 0 : ion_density_centered[0].size()) << std::endl;

    // Validate array dimensions
    assert(ion_density_centered.size() == local_NX);
    assert(!ion_density_centered.empty() && ion_density_centered[0].size() == local_NY + 2 * GHOST_CELLS);

    // Debug: Ensure no unexpected changes before proceeding
    if (rank == 0) {
        std::cout << "Before calling initializeSimulation:" << std::endl;
        std::cout << "ion_density_centered sizes: " << ion_density_centered.size() << " x " 
                  << ion_density_centered[0].size() << std::endl;
    }

    // Initialize simulation (Ensure this does not alter ion_density_centered sizes)
    double B0 = 5.0e-9;
    double dipole_moment = 7.8e22;
    double dipole_center_x = NX * DX / 2.0;
    double dipole_center_y = NY * DY / 2.0;
    double CURRENT_SHEET_Y = NY * DY / 2.0;
    double CURRENT_SHEET_THICKNESS = 1.0;
    initializeSimulation(local_NX, local_NY, rank, size, CURRENT_SHEET_Y, CURRENT_SHEET_THICKNESS, B0, dipole_center_x, dipole_center_y, dipole_moment);

    // Ensure no size changes post-initializeSimulation
    assert(ion_density_centered.size() == local_NX);
    assert(!ion_density_centered.empty() && ion_density_centered[0].size() == local_NY + 2 * GHOST_CELLS);

    // Step 3: Precompute WENO Weights
    int num_weights = local_NX - 2 * GHOST_CELLS;
    std::vector<WENOWeights> weights = precomputeWENOWeightsMPI(num_weights);

    // Step 4: Initialize velocity fields `vx` and `vy` with local dimensions
    std::vector<std::vector<double>> vx(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));
    std::vector<std::vector<double>> vy(local_NX, std::vector<double>(local_NY + 2 * GHOST_CELLS, 0.0));

    // Initialize velocities based on initial conditions
    for (int i = 0; i < local_NX; ++i) {
        for (int j = 0; j < local_NY + 2 * GHOST_CELLS; ++j) {
            vx[i][j] = 0.0; // Example: set initial velocity to zero
            vy[i][j] = 0.0; // Example: set initial velocity to zero
        }
    }

    // Step 5: Start the Time Loop
    timeLoop(weights, rank, size, px, py, vx, vy);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}