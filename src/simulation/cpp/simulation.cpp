/*
This C++ code generates simulated RF channels
*/

#include "simulation.h"

namespace py = pybind11;

std::vector<float> calculate_wavelengths_sim(float max_f, float min_f, uint num_samples) {
    /*
    Returns a vector of all wavelengths used in the FMCW chirp. 
    Parameters: 
        - max_f: Maximum frequency in hz
        - min_f: Minimum frequency in hz
        - num_samples: Number of samples
    Returns: vector of num_samples wavelengths from min_f to max_f
    */
    std::vector<float> wavelengths(num_samples);
    float step = (max_f - min_f) / num_samples;

    for (uint i = 0; i < num_samples; i++) {
        wavelengths[i] = SPEED_LIGHT / (min_f + i * step);
    }

    return wavelengths;
}

Simulation::Simulation(std::vector<Location> vert, std::vector<float> x_lim,   std::vector<float> y_lim,  std::vector<float> z_lim,  std::string radar_type) {
    /* Initialize Simulation
    Parameters:
        - vert: All verticies in mesh
        - x_lim/y_lim/z_lim: Limits of mesh in the X/Y/Z dimensions. Each one is a vector of 2 values (min, max)
        - radar_type: '77_ghz' or '24_ghz'. Changes which parameters to load from the params.json
    */

    // Load parameters from JSON
    auto res = get_radar_params(radar_type, "simulation", "", "../utils/params.json");
    this->min_f = std::get<0>(res);
    this->max_f = std::get<1>(res);
    this->slope = std::get<2>(res);
    this->num_samples = std::get<3>(res);
    this->wavelength = SPEED_LIGHT / this->max_f;
    this->wavelengths = calculate_wavelengths_sim(this->max_f, this->min_f, this->num_samples);

    this->x_lim = std::move(x_lim);
    this->y_lim = std::move(y_lim);
    this->z_lim = std::move(z_lim);
    this->simulated_channels.clear();    
	this->vert = std::move(vert);
}

void Simulation::simulate_rf_channel(std::vector<int> visible_vertices_index, Location this_radar_loc, Location initial_loc) {
    /*
    Simulates the RF channel when given which vertices in the mesh are visible, and the location of the radar.
    Parameters: 
        - visible_vertices_index: Contains the indices of the vertices which are "visible" (e.g., which vertices should be simulated)
        - this_radar_loc: The current location of the radar
        - initial_radar_loc: The starting location of the radar
    Returns: None
    */
    int vis_vert_size = visible_vertices_index.size();
    std::vector<float> distance(vis_vert_size);

    # pragma omp parallel for
    for (int i = 0; i < vis_vert_size; i++) {
        float x_diff = this->vert[visible_vertices_index[i]][0] - this_radar_loc[0];
        float y_diff = this->vert[visible_vertices_index[i]][1] - this_radar_loc[1];
        float z_diff = this->vert[visible_vertices_index[i]][2] - this_radar_loc[2];
		distance[i] = 2 * std::sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
    }

    Measurement sim_channels(this->wavelengths.size(), std::vector<std::complex<float>>(1)); //Simulation only has 1RX
	# pragma omp parallel for
    for (std::size_t j = 0; j < this->wavelengths.size(); j++) {
        sim_channels[j][0] = 0.0;
        for (std::size_t i = 0; i < distance.size(); i++) {
            sim_channels[j][0] += (1 / distance[i]) *
                std::exp(std::complex<float>(0, 2 * M_PI * (distance[i]) / this->wavelengths[j]));
        }
    }
    this->simulated_channels.push_back(sim_channels);

    Location shifted = { this_radar_loc[0] - initial_loc[0],
                         this_radar_loc[1] - initial_loc[1],
                         this_radar_loc[2] - initial_loc[2] };
    this->radar_locations.push_back(shifted);
}


std::vector<Measurement> Simulation::get_channels() {
    /*
    Returns the simulated RF channels. simulate_rf_channel must be called first to fill this vector.
    */
    return this->simulated_channels;
}

std::vector<Location> Simulation::get_shifted_radar_locations() {
    /*
    Returns the shifted (noisy) radar locations. simulate_rf_channel must be called first to fill this vector.
    */
    return this->radar_locations;
}


// Declare class for python
PYBIND11_MODULE(simulation, m) {  
	py::class_<Simulation>(m,"Simulation")
		.def(
			py::init< 
				std::vector<Location>, 
				std::vector<float>,
				std::vector<float>,
				std::vector<float>,
				std::string
			>()
		)
		.def("get_channels", &Simulation::get_channels)
		.def("get_shifted_radar_locations", &Simulation::get_shifted_radar_locations)
		.def("simulate_rf_channel", &Simulation::simulate_rf_channel);
}
