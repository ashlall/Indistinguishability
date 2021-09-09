#ifndef READ_WRITE_H
#define READ_WRITE_H

#include "data_struct.h"
#include "data_utility.h"
#include "operation.h"


// read points from the input file
point_set_t* read_points(char* input);

// check domination
int dominates(point_t* p1, point_t* p2);

// check c-domination
int c_dominates(double c, point_t* p1, point_t* p2);

// compute the c-skyline set
point_set_t* c_skyline_point(double c, point_set_t *p);

// prepare the file for computing the convex hull (the candidate utility range R) via half space interaction
void write_hyperplanes(vector<hyperplane_t*> utility_hyperplane, point_t* feasible_pt, char* filename);


#endif
