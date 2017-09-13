/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);


  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = particles.size();
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  } 

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  
  for (auto& particle : particles) {

    auto x = fabs(yaw_rate) <= 0.00001 ? particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) :
                                         particle.x + velocity * cos(particle.theta) * delta_t;
    auto y = fabs(yaw_rate) <= 0.00001 ? particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) :
                                         particle.y + velocity * sin(particle.theta) * delta_t;
    auto theta = particle.theta + yaw_rate*delta_t;

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto& observation : observations) {
    auto landmarkIt = predicted.begin();
    auto endIt = predicted.end();
    if (landmarkIt != endIt) {
      auto min_d = dist(landmarkIt->x, landmarkIt->y, observation.x, observation.y);
      observation.id = landmarkIt->id;
      while (++landmarkIt != endIt) {
        auto d = dist(landmarkIt->x, landmarkIt->y, observation.x, observation.y);
        if (d < min_d) {
          min_d = d;
          observation.id = landmarkIt->id;
        }
      }
    }
  }  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  double particle_weight_sum = 0.0;
  for (auto& particle : particles) {
    
    std::vector<LandmarkObs> observationsCpy = observations;

    //Transform to map coordinate system
    for (auto& observation : observationsCpy) {
      ToMapCoordinates(particle.x, particle.y, particle.theta, observation.x, observation.y);
    }

    //Find predicted landmarks within range of sensor
    std::vector<LandmarkObs> predictedLandmarks;
    for (auto& mapPoint : map_landmarks.landmark_list) {
      auto distance = dist(particle.x, particle.y, mapPoint.x_f, mapPoint.y_f);
      if (distance <= sensor_range) {
        LandmarkObs predictedLandmark(mapPoint.id_i, mapPoint.x_f, mapPoint.y_f);
        predictedLandmarks.push_back(predictedLandmark);
      }
    }
    dataAssociation(predictedLandmarks, observationsCpy);

    particle.weight = 1;
    for (auto& observation : observationsCpy) {
      auto landmarkIt = std::find_if(predictedLandmarks.begin(), predictedLandmarks.end(), [&observation](LandmarkObs& landmark)->bool { return landmark.id == observation.id; });
      particle.weight *= MulVarDensity(observation.x, observation.y, landmarkIt->x, landmarkIt->y, std_x, std_y);
    }
    
    particle_weight_sum += particle.weight;
  }

  //Normalize the weights.
  std::for_each(particles.begin(), particles.end(), [particle_weight_sum](Particle& particle) { particle.weight /= particle_weight_sum; });
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<double> weights(particles.size());
  std::transform(particles.begin(), particles.end(), weights.begin(), [](Particle& particle) { return particle.weight; });

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());

  std::vector<Particle> newParticleList;
  for (int i = 0, count = particles.size(); i < count; ++i) {
    newParticleList.push_back(particles[d(gen)]);
  }

  particles = newParticleList;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
