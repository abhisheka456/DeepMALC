#include <map> 
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;


void generate_offsets_cy(int d,
                         int base,
                         int * offsets) {
    /*
        Generate 3**d neighbors for any point.

        Parameters
        ----------
        d: Dimensions
        base: 3, corresponding to (-1, 0, 1)
        offsets: (3**d, d) array of offsets to be added to 
                 a bin to get neighbors

    */

    int tmp_i;

    for (int i = 0; i < pow(base, d); i++) {
        tmp_i = i;
        for (int j = 0; j < d; j++) {
            if (tmp_i == 0) break;
            offsets[i * d + j] = tmp_i % base - 1;
            tmp_i /= base;
        }
    }
}


void grid_cluster(int n,
                  int d,
                  int base,
                  int iterations,
                  float bandwidth,
                  int * offsets,
                  float * X_shifted,
                  int * membership, 
                  int * k_num) {
                  
    map< vector<int>, pair< vector<float>, int> > cluster_grid;
    map< vector<int>, int > map_cluster;
    map< int, int > clus;
    map< int, int > :: iterator it2;
    map< vector<int>, pair< vector<float>, int> >:: iterator it;
    map< vector<int>, pair< vector<float>, int> > means;
    map< vector<int>, vector<int> > element_grid;

    int iter = 0;
    vector<int> current_bin(d);
    vector<int> bin(d);
    vector<int> membershipp(n);
    vector<int> membershipp_old(n);

    // new clustering at grids
    
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            bin[k] = X_shifted[i * d + k] / bandwidth;
        }
        if (cluster_grid.find(bin) == cluster_grid.end()) {
            cluster_grid[bin] = make_pair(std::vector<float>(d, 0), 0);
        }
        for (int k = 0; k < d; k++){
            cluster_grid[bin].first[k] += X_shifted[i * d + k];
        }
        cluster_grid[bin].second++;
        if (element_grid.find(bin) == element_grid.end()) {
            element_grid[bin] = vector<int>(1,i);
        } else {
            element_grid[bin].push_back(i);
        }
    }
    
    
    //copy(membershipp.begin(), membershipp.end(), membership);              
    while (iter <= iterations){
        iter++;
        means.clear();
        int temp = 0;
        for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){
        
            for (int j = 0; j < pow(base, d); j++) {
            
                for (int k = 0; k < d; k++) {
                
                    current_bin[k] = it->first[k] + offsets[j * d + k];
                    
                    if (j == 0){
                    
                        bin[k] =  it->first[k] ;
                        
                    }
                     
                 
                }
            
                // If neighbor exists, add it to the mean
                if (cluster_grid.find(current_bin) != cluster_grid.end()) {
                    temp++;
                    if (means.find(current_bin) == means.end()) {
                        means[current_bin] = make_pair(std::vector<float>(d, 0), 0);
                    }
                
                    for (int k = 0; k < d; k++) {
                        means[current_bin].first[k] += cluster_grid[bin].first[k] * 1.0;
                    }

                    means[current_bin].second += cluster_grid[bin].second;
                    }
                }
            
            }
        
         for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){
            for (int k = 0; k < d; k++) {
                current_bin[k] = it->first[k];
            }
        
            for (int k = 0; k < d; k++) {
                cluster_grid[current_bin].first[k] = means[current_bin].first[k] * 1.0 / means[current_bin].second;
            }
        
        }
       
        // update cluster grid and membership 
        map< vector<int>, pair< vector<float>, int> > cluster_grid_old = cluster_grid;
        map< vector<int>, vector<int> > element_grid_old = element_grid;
        element_grid.clear();
        cluster_grid.clear();

        


        for (it = cluster_grid_old.begin(); it != cluster_grid_old.end(); ++it ){

            for (int k = 0; k < d; k++) {
                bin[k] = it->second.first[k] / bandwidth;
                current_bin[k] = it->first[k];
            }
            
            if (cluster_grid.find(bin) == cluster_grid.end()) {
                cluster_grid[bin] = make_pair(std::vector<float>(d, 0), 0);
            }   
            
            for (int k = 0; k < d; k++){
                cluster_grid[bin].first[k] += it->second.first[k] * 1.0 * it->second.second;
            }
            cluster_grid[bin].second += it->second.second;
            
            if (element_grid.find(bin) == element_grid.end()) {
                element_grid[bin] = element_grid_old[current_bin];
            } else {

                element_grid[bin].insert( element_grid[bin].end(), element_grid_old[current_bin].begin(), element_grid_old[current_bin].end() );
            }
        }
        cluster_grid_old.clear();
        element_grid_old.clear();

        if (temp == 0){
            break;
        }
    
    }
    int temp = 0;
    vector<int> k_num2(1);
    k_num2[0] = cluster_grid.size();
    vector<float> bins(k_num2[0] * d);
    for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){
        for (int k = 0; k < d; k++) {
            bins[temp * d + k] = it->second.first[k] *1.0 / it->second.second;

        }
        bin = it->first;
        for(int i = 0; i < element_grid[bin].size(); i++){
            membershipp[element_grid[bin][i]] = temp;
        }
        temp++;
    }
    
    
    copy(membershipp.begin(), membershipp.end(), membership);
    copy(bins.begin(), bins.end(),X_shifted);  
    copy(k_num2.begin(), k_num2.end(),k_num); 
}
