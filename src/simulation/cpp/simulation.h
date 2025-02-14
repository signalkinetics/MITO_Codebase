#include "mmwave_common.h"


class Simulation {
	public:
		std::vector<float> x_lim;
		std::vector<float> y_lim;
		std::vector<float> z_lim;
		float min_f;
		float max_f;
		uint num_samples;

		std::vector<Location> vert;
		std::vector<Location> radar_locations;
		std::vector<Measurement> simulated_channels;
		std::vector<float> wavelengths;

		float wavelength;
		float slope;


		Simulation(
			std::vector<Location>, 
			std::vector<float>, 
			std::vector<float>, 
			std::vector<float>, 
			std::string 
		);
		void simulate_rf_channel(std::vector<int>, Location, Location);
		Image image_sim();
		std::vector<Measurement> get_channels();
		std::vector<Location> get_shifted_radar_locations();
};
