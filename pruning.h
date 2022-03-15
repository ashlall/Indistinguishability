#ifndef PRUNING_H
#define PRUNING_H

#include "data_struct.h"
#include "data_utility.h"

#include "operation.h"
#include "lp.h"
#include "rtree.h"
#include "frame.h"
#include "read_write.h"
#include <queue>

// the domination options
#define HYPER_PLANE 1
#define CONICAL_HULL 2

// the skyline options
#define SQL 1
#define RTREE 2

//  the stopping options
#define NO_BOUND 1
#define EXACT_BOUND 2
#define APPROX_BOUND 3

using namespace std;

// get the set of extreme points of the candidate utility range R (bounded by the extreme vectors)
vector<point_t*> get_extreme_pts(vector<point_t*>& ext_vec);

// use the seqentail way for maintaining the candidate set
void sql_pruning(point_set_t* P, vector<int>& C_idx, vector<point_t*>& ext_vec, double& rr, int stop_option, int dom_option, double epsilon);

// use the branch-and-bound skyline (BBS) algorithm for maintaining the candidate set
void rtree_pruning(point_set_t* P, vector<int>& C_idx, vector<point_t*>& ext_vec, double& rr,  int stop_option, int dom_option, double epsilon);


void get_hyperplanes(vector<point_t*>& ext_vec, hyperplane_t*& hp, vector<point_t*>& hyperplanes);
double get_rrbound_exact(vector<point_t*> ext_pts);
double get_rrbound_approx(vector<point_t*> ext_pts);
int hyperplane_dom(point_t* p_i, point_t* p_j, vector<point_t*> ext_pts, double epsilon);
int conical_hull_dom(point_t* p_i, point_t* p_j, hyperplane_t* hp, vector<point_t*> hyperplanes, vector<point_t*> ext_vec, double epsilon);
int dom(point_t* p_i, point_t* p_j, vector<point_t*> ext_pts, hyperplane_t* hp, vector<point_t*> hyperplanes, vector<point_t*> ext_vec, int dom_option, double epsilon);

// Compute the width of the feasible region for the MinR algorithm
double get_R_width(vector<point_t*> ext_pts);

// Compute the diameter of the feasible region for the MinD algorithm
double get_R_diameter(vector<point_t*> ext_pts);

#endif
