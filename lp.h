#ifndef LP_H
#define LP_H

#include "glpk.h"
#include "data_struct.h"
#include "data_utility.h"
#include "operation.h"
#include <vector>

// solve LP using GLPK

// Use LP to check whehter a point pt is a conical combination of the vectors in ExRays
bool insideCone(std::vector<point_t*> ExRays, point_t* pt);

// Use LP to find a feasible point of the half sapce intersection (used later in Qhull for half space intersection)
point_t* find_feasible(std::vector<hyperplane_t*> hyperplane);

// solve the LP in frame computation
void solveLP(std::vector<point_t*> B, point_t* b, double& theta, point_t* & pi);

#endif
