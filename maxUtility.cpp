#include "maxUtility.h"

// get the index of the "current best" point
// P: the input car set
// C_idx: the indexes of the current candidate favorite car in P
// ext_vec: the set of extreme vecotr
int get_current_best_pt(point_set_t* P, vector<int>& C_idx, vector<point_t*>& ext_vec)
{
	int dim = P->points[0]->dim;

	// the set of extreme points of the candidate utility range R
	vector<point_t*> ext_pts;
	ext_pts = get_extreme_pts(ext_vec);

	// use the "mean" utility vector in R (other strategies could also be used)
	point_t* mean = alloc_point(dim);
	for(int i = 0; i < dim; i++)
	{
		mean->coord[i] = 0;
	}
	for(int i = 0; i < ext_pts.size(); i++)
	{
		for(int j = 0; j < dim; j++)
			mean->coord[j] += ext_pts[i]->coord[j];
	}
	for(int i = 0; i < dim; i++)
	{
		mean->coord[i] /= ext_pts.size();
	}

	// look for the maximum utility point w.r.t. the "mean" utility vector
	int best_pt_idx;
	double max = 0;
	for(int i = 0; i < C_idx.size(); i++)
	{
		point_t* pt = P->points[C_idx[i]];

		double v = dot_prod(pt, mean);
		if(v > max)
		{
			max = v;
			best_pt_idx =  C_idx[i];
		}
	}

	release_point(mean);

	for(int i = 0; i < ext_pts.size(); i++)
		release_point(ext_pts[i]);
	return best_pt_idx;
}

// generate s cars for selection in a round
// P: the input car set
// C_idx: the indexes of the current candidate favorite car in P
// s: the number of cars for user selection
// ext_vec: Current bounds on utility function; needed for repeated random selection
// frame: the frame for obtaining the set of neibouring vertices of the current best vertiex (used only if cmp_option = SIMPLEX)
// cmp_option: the car selection mode, which must be either SIMPLEX or RANDOM
// prune_option: Needed for the repeated random selection technique
// epsilon: the desired indistinguishability threshold
// delta: the indistinguishability threshold for user selection
vector<int> generate_S(point_set_t* P, vector<int>& C_idx, int s, vector<point_t*> ext_vec,  int current_best_idx, int& last_best, vector<int>& frame, int cmp_option, int stop_option, int prune_option, int dom_option, double epsilon, double delta, int repeats)
{
	// the set of s cars for selection
        vector<int> S;
	vector<int> C_idx_backup;
	vector<point_t*> ext_vec_backup;
	

	if(cmp_option == RANDOM) // RANDOM car selection mode
	{
		// randoming select at most s non-overlaping cars in the candidate set 
		while(S.size() < s && S.size() < C_idx.size())
		{
			int idx = rand() % C_idx.size();

			bool isNew = true;
			for(int i = 0; i < S.size(); i++)
			{
				if(S[i] == idx)
				{
					isNew = false;
					break;
				}
			}
			if(isNew)
				S.push_back(idx);
		}
	}
	else if(cmp_option == SIMPLEX) // SIMPLEX car selection mode
	{
		if(last_best != current_best_idx || frame.size() == 0) // the new frame is not computed before (avoid duplicate frame computation)
		{
			// create one ray for each car in P for computing the frame
			vector<point_t*> rays;
			int best_i = -1;
			for(int i = 0; i < P->numberOfPoints; i++)
			{
				if(i == current_best_idx)
				{
					best_i = i;
					continue;
				}

				point_t* best = P->points[current_best_idx];
				point_t* newRay = sub(P->points[i], best);
				rays.push_back(newRay);
			}

			// frame compuatation
			frameConeFastLP(rays, frame);
		
			// update the indexes lying after current_best_idx
			for(int i = 0; i < frame.size(); i++)
			{
				if(frame[i] >= current_best_idx)
					frame[i]++;

				//S[i] = C_idx[S[i]];
			}

			for(int i = 0; i < rays.size(); i++)
				release_point(rays[i]);
		}

		//printf("current_best: %d, frame:", P->points[current_best_idx]->id);
		//for(int i = 0; i < frame.size(); i++)
		//	printf("%d ", P->points[frame[i]]->id);
		//printf("\n");

		//S.push_back(best_i);

		for(int j = 0; j < C_idx.size(); j++)
		{
			if(C_idx[j] == current_best_idx) // it is possible that current_best_idx is no longer in the candidate set, then no need to compare again
			{
				S.push_back(j);
				break;
			}
		}

		// select at most s non-overlaping cars in the candidate set based on "neighboring vertices" obtained via frame compuation
		for(int i = 0; i < frame.size() && S.size() < s; i++)
		{
			for(int j = 0; j < C_idx.size() && S.size() < s; j++)
			{
				if(C_idx[j] == current_best_idx)
					continue;

				if(C_idx[j] == frame[i])
					S.push_back(j);
			}
		}

		// if less than s car are selected, fill in the remaing one
		if (S.size() < s && C_idx.size() > s)
		{
			for (int j = 0; j < C_idx.size(); j++)
			{
				bool noIn = true;
				for (int i = 0; i < S.size(); i++)
				{
					if (j == S[i])
						noIn = false;
				}
				if (noIn)
					S.push_back(j);

				if (S.size() == s)
					break;
			}
		}
	}
	else if (cmp_option == MINC)
	{
	  double rr = 1;
	  vector<int> best_S;
	  double best_value = C_idx.size() + 1;

	  vector<point_t*> ext_pts;
	  vector<point_t*> hyperplanes;
	  hyperplane_t* hp = NULL;

	  if(dom_option == HYPER_PLANE)
	    ext_pts = get_extreme_pts(ext_vec); // in Hyperplane Pruning, we need the set of extreme points of R
	  else
	    {
	      // in Conical Pruning, we need bounding hyperplanes for the conical hull
	      get_hyperplanes(ext_vec, hp, hyperplanes);
	      if(stop_option != NO_BOUND) // if an upper bound on the regret ratio is needed, we need the set of extreme points of R
		ext_pts = get_extreme_pts(ext_vec);
	    }


	  if (ext_pts.size() == 0)
       	    cout << "ext_pts has size 0 at start of MINC --- problematic" << endl;

	  //cout << "computed ext_pts of size " << ext_pts.size() << endl;	    

	  // the upper bound of the regret ratio based on (the extreme ponits of) R
	  rr = 1;
	  
	  // run the adapted squential skyline algorihtm
	  int* sl = new int[C_idx.size()]; // this will hold the indexes of C_idx that are on the skyline
	  int sl_size = 0;
	  for (int i = 0; i < C_idx.size(); ++i)
	    {
	      int dominated = 0;
	      point_t* pt = P->points[C_idx[i]];

	      // check if pt is dominated by the skyline so far
	      for (int j = 0; j < sl_size && !dominated; ++j)
                {
                  if(dom(P->points[ C_idx[ sl[j] ] ], pt, ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
		    dominated = 1;
                }

	      if (!dominated)
                {
		  // eliminate any points in current skyline that it dominates
		  int m = sl_size;
		  sl_size = 0;
		  for (int j = 0; j < m; ++j)
		    {

		      if(!dom(pt, P->points[ C_idx[ sl[j] ] ], ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
			sl[sl_size++] = sl[j];
		    }
		  
		  // add this point as well
		  sl[sl_size++] = i;
                }
	    }

	  //printf("Skyline: %d points\n", sl_size);
	  //for(int i = 0; i < sl_size; ++i)
	  //  print_point(P->points[ C_idx [ sl[i]] ]);


	  // try 'repeats' random options
	  for (int rep = 0; rep < repeats; ++rep)
	  {
	    // randomly select at most s non-overlaping cars in the skyline
	    
	    S.clear();
	    while(S.size() < s && S.size() < sl_size)
	    {
	      int idx = rand() % sl_size; //C_idx.size();
	      
	      bool isNew = true;
	      for(int i = 0; i < S.size(); i++)
	      {
		if(S[i] == sl[idx]) // || dominates(P->points[ C_idx[ S[i] ] ], P->points[ C_idx[idx] ]) || dominates(P->points[ C_idx[ idx ] ], P->points[ C_idx[ S[i] ] ]))
		{
		  isNew = false;
		  break;
		}
	      }
	      if(isNew)
		S.push_back(sl[idx]);
	    }

	    vector<point_t*> S_real(S.size());
	    for(int i = 0; i < S.size(); ++i)
	      S_real[i] = P->points[ C_idx[ S[i] ] ];
	    

	    // if there are fewer than s points in the skyline, just return them
	    if (S.size() < s)
	    {
	      delete[] sl;
	      return S;
	    }

	    // see how many points don't get pruned for each option and keep track of the most left
            size_t most_left = 0;
	    double avg_left = 0.0;
	    	    
	    //cout << "s = " << s << endl;
	    //cout << "Points in S: " << endl;
	    //for(int j = 0; j < s; ++j)
	    //  print_point(P->points[ S[j] ]);
	    //cout << "Points in ext_vec:" << endl;
	    //for (int j = 0; j < ext_vec.size(); ++j)
	    //  print_point(ext_vec[j]);
	    //cout << "Testing " << repeats << endl;
	    
	    for (int max_i = 0; max_i < s; ++max_i)
	    {
	      // back up variables
	      C_idx_backup = C_idx;
	      ext_vec_backup = ext_vec;

	      // update the extreme vectors based on the user feedback of max_i
	      update_ext_vec(P, S_real, C_idx_backup, max_i, s, ext_vec_backup, current_best_idx, last_best, frame, cmp_option, delta);

	      //update candidate set
	      if(prune_option == SQL)
	        sql_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);
	      else
	        rtree_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);

	      // calculate number of points eliminated and update min
              if (C_idx_backup.size() > most_left)
                most_left = C_idx_backup.size();

	      // calculate the average left, too
	      avg_left += most_left / s;

	      //cout << C_idx_backup.size() << " " << most_left << endl;

	      // release newly created points in ext_vec_backup
	      for (int j = ext_vec.size(); j < ext_vec_backup.size(); ++j)
	      	release_point(ext_vec_backup[j]);
	    }
	    
	    //cout << "most left = " << most_left << endl;
	    
	    if (avg_left < best_value) //(most_left < best_value)
              {
                best_S = S;
                best_value = avg_left; //most_left;
              }
	  }

	  delete[] sl;
	  //cout << "best value = " << best_value << endl;	  
	  S = best_S;
	  //cout << "S:\n";
	  //for(int j = 0; j < S.size(); ++j)
	  //  print_point(P->points[ C_idx[ S[j] ] ]);
	}
	else if (cmp_option == MINR)
	  {
	    double rr = 1;
	    vector<int> best_S;
	    
	    vector<point_t*> ext_pts;
	    vector<point_t*> hyperplanes;
	    hyperplane_t* hp = NULL;

	    if(dom_option == HYPER_PLANE)
	      ext_pts = get_extreme_pts(ext_vec); // in Hyperplane Pruning, we need the set of extreme points of R
	    else
	      {
		// in Conical Pruning, we need bounding hyperplanes for the conical hull
		get_hyperplanes(ext_vec, hp, hyperplanes);
		if(stop_option != NO_BOUND) // if an upper bound on the regret ratio is needed, we need the set of extreme points of R
		  ext_pts = get_extreme_pts(ext_vec);
	      }

	    if (ext_pts.size() == 0)
	      {
		cout << "ext_pts has size 0 at start of MINR --- problematic" << endl;
		cout << "Ext vec:" << endl;
		for(int i = 0; i < ext_vec.size(); ++i)
		  print_point(ext_vec[i]);
	      }

	    //cout << "computed ext_pts of size " << ext_pts.size() << endl;

	    // the upper bound of the regret ratio based on (the extreme ponits of) R
	    rr = 1;

	    // run the adapted squential skyline algorihtm
	    int* sl = new int[C_idx.size()]; // this will hold the indexes of C_idx that are on the skyline
	    int sl_size = 0;
	    for (int i = 0; i < C_idx.size(); ++i)
	      {
		int dominated = 0;
		point_t* pt = P->points[C_idx[i]];

		// check if pt is dominated by the skyline so far
		for (int j = 0; j < sl_size && !dominated; ++j)
		  {
		    if(dom(P->points[ C_idx[ sl[j] ] ], pt, ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
		      dominated = 1;
		  }

		if (!dominated)
		  {
		    // eliminate any points in current skyline that it dominates
		    int m = sl_size;
		    sl_size = 0;
		    for (int j = 0; j < m; ++j)
		      {

			if(!dom(pt, P->points[ C_idx[ sl[j] ] ], ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
			  sl[sl_size++] = sl[j];
		      }	      

		    // add this point as well
		    sl[sl_size++] = i;
		  }
	      }

	    //printf("Skyline: %d points\n", sl_size);
	    //for(int i = 0; i < sl_size; ++i)
	    //  print_point(P->points[ C_idx [ sl[i]] ]);


	    double best_R_width = get_R_width(ext_pts) * 2;
	    //cout << "initial R width = " << best_R_width << endl;

	    // try 'repeats' random options
	    for (int rep = 0; rep < repeats; ++rep)
	      {
		// randomly select at most s non-overlaping cars in the skyline
		S.clear();
		while(S.size() < s && S.size() < sl_size) //C_idx.size())
		  {
		    int idx = rand() % sl_size; //C_idx.size();

		    bool isNew = true;
		    for(int i = 0; i < S.size(); i++)
		      {
			if(S[i] == sl[idx]) // C_idx[idx])
			  {
			    isNew = false;
			    break;
			  }
		      }
		    if(isNew)
		      S.push_back(sl[idx]); //idx);
		  }

		//cout << "S:\n";
		//for(int j = 0; j < S.size(); ++j)
		//  print_point(P->points[ C_idx[ S[j] ] ]);


		// if there are fewer than s points in the skyline, just return them
		if (S.size() < s)
		{
		  delete[] sl;
		  return S;
		}

		//cout << "s = " << s << endl;
		//cout << "Points in S: " << endl;
		//for(int j = 0; j < s; ++j)
		//  print_point(P->points[ S[j] ]);
		//cout << "Points in ext_vec:" << endl;
		//for (int j = 0; j < ext_vec.size(); ++j)
		//  print_point(ext_vec[j]);
		//cout << "Testing " << repeats << endl;


		vector<point_t*> S_real(S.size());
		for(int i = 0; i < S.size(); ++i)
		  S_real[i] = P->points[ C_idx[ S[i] ] ];

		double worst_R_width = 0.0;
		double avg_R_width = 0.0;

		for (int max_i = 0; max_i < S.size(); ++max_i)
		{
		  // back up variables
		  C_idx_backup = C_idx;
		  ext_vec_backup = ext_vec;
		  
		  // generate the options for user selection and update the extreme vectors based on the user feedback
		  update_ext_vec(P, S_real, C_idx_backup, max_i, s, ext_vec_backup, current_best_idx, last_best, frame, cmp_option, delta);
		  
		  //DON'T update candidate set 
		  //if(prune_option == SQL)
		  //  sql_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);
		  //else
		  //  rtree_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);
		  
		  
		  vector<point_t*> new_ext_pts = get_extreme_pts(ext_vec_backup);
		  double R_width = get_R_width(new_ext_pts);
		  
		  // calculate number of points eliminated and update min
		  if (R_width > worst_R_width)
		    worst_R_width = R_width;
		  
		  // calculate average as well
		  avg_R_width += R_width / s;

		  //cout << C_idx_backup.size() << " " << most_left << endl;
		  
		  // release newly created points in ext_vec_backup
		  for (int j = ext_vec.size(); j < ext_vec_backup.size(); ++j)
		    release_point(ext_vec_backup[j]);

		  // release points in new_ext_pts
		  for(int j = 0; j < new_ext_pts.size(); ++j)
		    release_point(new_ext_pts[j]);
		}
		
		if (avg_R_width < best_R_width) //(worst_R_width < best_R_width)
		{
		  best_S = S;
		  best_R_width = avg_R_width; //worst_R_width;
		}
		
		//cout << "Best R width : " << best_R_width << endl;
		
	      }
	    
	    delete[] sl;
	    S = best_S;
	    //cout << "Best width = " << best_R_width << endl;
	    //cout << "S has " << S.size() << " elements\n";
	    //for(int j = 0; j < S.size(); ++j)
	    //  print_point(P->points[ C_idx[ S[j] ] ]);
	  }
	else if (cmp_option == MIND)
          {
            double rr = 1;
            vector<int> best_S;

            vector<point_t*> ext_pts;
            vector<point_t*> hyperplanes;
            hyperplane_t* hp = NULL;

            if(dom_option == HYPER_PLANE)
              ext_pts = get_extreme_pts(ext_vec); // in Hyperplane Pruning, we need the set of extreme points of R
            else
              {
                // in Conical Pruning, we need bounding hyperplanes for the conical hull
                get_hyperplanes(ext_vec, hp, hyperplanes);
                if(stop_option != NO_BOUND) // if an upper bound on the regret ratio is needed, we need the set of extreme points of R
                  ext_pts = get_extreme_pts(ext_vec);
              }

            if (ext_pts.size() == 0)
              {
                cout << "ext_pts has size 0 at start of MIND --- problematic" << endl;
                cout << "Ext vec:" << endl;
                for(int i = 0; i < ext_vec.size(); ++i)
                  print_point(ext_vec[i]);
              }

            //cout << "computed ext_pts of size " << ext_pts.size() << endl;

            // the upper bound of the regret ratio based on (the extreme ponits of) R
            rr = 1;

	    // run the adapted squential skyline algorihtm
            int* sl = new int[C_idx.size()]; // this will hold the indexes of C_idx that are on the skyline
            int sl_size = 0;
            for (int i = 0; i < C_idx.size(); ++i)
              {
                int dominated = 0;
                point_t* pt = P->points[C_idx[i]];

                // check if pt is dominated by the skyline so far
                for (int j = 0; j < sl_size && !dominated; ++j)
                  {
                    if(dom(P->points[ C_idx[ sl[j] ] ], pt, ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
                      dominated = 1;
                  }

                if (!dominated)
                  {
                    // eliminate any points in current skyline that it dominates
                    int m = sl_size;
                    sl_size = 0;
                    for (int j = 0; j < m; ++j)
                      {

                        if(!dom(pt, P->points[ C_idx[ sl[j] ] ], ext_pts, hp, hyperplanes, ext_vec, dom_option, 0))
                          sl[sl_size++] = sl[j];
                      }

                    // add this point as well
                    sl[sl_size++] = i;
                  }
              }

            //printf("Skyline: %d points\n", sl_size);
            //for(int i = 0; i < sl_size; ++i)
            //  print_point(P->points[ C_idx [ sl[i]] ]);

	    double best_R_diameter = get_R_diameter(ext_pts) * 2;
            //cout << "initial R diameter = " << best_R_diameter << endl;

            // try 'repeats' random options
            for (int rep = 0; rep < repeats; ++rep)
              {
                // randomly select at most s non-overlaping cars in the skyline
                S.clear();
                while(S.size() < s && S.size() < sl_size) //C_idx.size())
                  {
                    int idx = rand() % sl_size; //C_idx.size();

                    bool isNew = true;
                    for(int i = 0; i < S.size(); i++)
                      {
                        if(S[i] == sl[idx]) // C_idx[idx])
                          {
                            isNew = false;
                            break;
                          }
                      }
                    if(isNew)
                      S.push_back(sl[idx]); //idx);
                  }

                //cout << "S:\n";
                //for(int j = 0; j < S.size(); ++j)
                //  print_point(P->points[ C_idx[ S[j] ] ]);


                // if there are fewer than s points in the skyline, just return them
                if (S.size() < s)
		  {
		    delete[] sl;
		    return S;
		  }

		//cout << "s = " << s << endl;
                //cout << "Points in S: " << endl;
                //for(int j = 0; j < s; ++j)
                //  print_point(P->points[ S[j] ]);
                //cout << "Points in ext_vec:" << endl;
                //for (int j = 0; j < ext_vec.size(); ++j)
                //  print_point(ext_vec[j]);
                //cout << "Testing " << repeats << endl;


                vector<point_t*> S_real(S.size());
                for(int i = 0; i < S.size(); ++i)
                  S_real[i] = P->points[ C_idx[ S[i] ] ];

                double worst_R_diameter = 0.0;
                double avg_R_diameter = 0.0;

                for (int max_i = 0; max_i < S.size(); ++max_i)
		  {
		    // back up variables
		    C_idx_backup = C_idx;
		    ext_vec_backup = ext_vec;

		    // generate the options for user selection and update the extreme vectors based on the user feedback
		    update_ext_vec(P, S_real, C_idx_backup, max_i, s, ext_vec_backup, current_best_idx, last_best, frame, cmp_option, delta);

		    //DON'T update candidate set
		    //if(prune_option == SQL)
		    //  sql_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);
		    //else
		    //  rtree_pruning(P, C_idx_backup, ext_vec_backup, rr, stop_option, dom_option, epsilon);


		    vector<point_t*> new_ext_pts = get_extreme_pts(ext_vec_backup);
		    double R_diameter = get_R_diameter(new_ext_pts);

		    // calculate number of points eliminated and update min
		    if (R_diameter > worst_R_diameter)
		      worst_R_diameter = R_diameter;

		    // calculate average as well
		    avg_R_diameter += R_diameter / s;

		    //cout << C_idx_backup.size() << " " << most_left << endl;

		    // release newly created points in ext_vec_backup
		    for (int j = ext_vec.size(); j < ext_vec_backup.size(); ++j)
		      release_point(ext_vec_backup[j]);

		    // release points in new_ext_pts
		    for(int j = 0; j < new_ext_pts.size(); ++j)
		      release_point(new_ext_pts[j]);
		  }

                if (avg_R_diameter < best_R_diameter) //(worst_R_diameter < best_R_diameter)
		  {
		    best_S = S;
		    best_R_diameter = avg_R_diameter; //worst_R_diameter;
		  }

                //cout << "Best R diameter : " << best_R_diameter << endl;

              }

            delete[] sl;
            S = best_S;
            //cout << "Best diameter = " << best_R_diameter << endl;
            //cout << "S has " << S.size() << " elements\n";
            //for(int j = 0; j < S.size(); ++j)
            //  print_point(P->points[ C_idx[ S[j] ] ]);
	}
	else // for testing only. Do not use this!
	{
		vector<point_t*> rays;

		int best_i = -1;
		for(int i = 0; i < C_idx.size(); i++)
		{
			if(C_idx[i] == current_best_idx)
			{
				best_i = i;
				continue;
			}

			point_t* best = P->points[current_best_idx];

			point_t* newRay = sub(P->points[C_idx[i]], best);

			rays.push_back(newRay);
		}

		partialConeFastLP(rays, S, s - 1);
		if(S.size() > s - 1)
			S.resize(s - 1);
		for(int i = 0; i < S.size(); i++)
		{
			if(S[i] >= best_i)
				S[i]++;

			//S[i] = C_idx[S[i]];
		}
		S.push_back(best_i);


		for(int i = 0; i < rays.size(); i++)
			release_point(rays[i]);
	}
	return S;
}

// generate the options for user selection and update the extreme vecotrs based on the user feedback
// wPrt: record user's feedback
// P_car: the set of candidate cars with seqential ids
// S: The subset of tuples shown to the user
// skyline_proc_P: the skyline set of normalized cars
// C_idx: the indexes of the pruned approximation of I
// max_i: The index of the item in S the user prefers the most
// ext_vec: the set of extreme vecotr
// current_best_idx: the current best car
// last_best: the best car in previous interaction
// frame: the frame for obtaining the set of neibouring vertices of the current best vertiex (used only if cmp_option = SIMPLEX)
// cmp_option: the car selection mode, which must be either SIMPLEX or RANDOM
// delta: the amount of error (fraction) possible when a user makes a selection
void update_ext_vec(point_set_t* P, vector<point_t*> S, vector<int>& C_idx, int max_i, int s, vector<point_t*>& ext_vec, int& current_best_idx, int& last_best, vector<int>& frame, int cmp_option, double delta)
{
	// for each non-favorite car, create a new extreme vecotr
	for(int i = 0; i < S.size(); i++)
	{
		if(max_i == i)
			continue;

		point_t* scaled = scale(1 + delta, S[max_i]);
		point_t* tmp = sub(S[i], scaled); 
		//C_idx[S[i]] = -1;  // we can't necessarily eliminate this point

		point_t* new_ext_vec = scale(1 / calc_len(tmp), tmp);
		
		release_point(tmp);
		release_point(scaled);
		ext_vec.push_back(new_ext_vec);
	}

	// directly remove the non-favorite car from the candidate set
	//vector<int> newC_idx;
	//for(int i = 0; i < C_idx.size(); i++)
	//{
	//	if(C_idx[i] >= 0)
	//		newC_idx.push_back(C_idx[i]);
	//}
	//C_idx = newC_idx;
}

// the main interactive algorithm for real points
// P: the input dataset (assumed skyline)
// u: the unkonwn utility vector
// s: the question size
// epsilon: the desired indistinguishability threshold 
// delta: the indistinguishability threshold for user selection
// maxRound: the maximum number of rounds of interacitons
// Qcount: the number of question asked
// Csize: the size the candidate set when finished
// cmp_option: the car selection mode, which must be either SIMPLEX or RANDOM
// stop_option: the stopping condition, which must be NO_BOUND or EXACT_BOUND or APRROX_BOUND
// prune_option: the skyline algorithm, which must be either SQL or RTREE
// dom_option: the domination checking mode, which must be either HYPER_PLANE or CONICAL_HULL
// number of times to perform repeats for MINR and MINC algorithms
double max_utility(point_set_t* P, point_t* u, int s,  double epsilon, double delta, int maxRound, int &Qcount, int &Csize,  int cmp_option, int stop_option, int prune_option, int dom_option, int repeats)
{	
	int dim = P->points[0]->dim;

	// the indexes of the candidate set
	// initially, it is all the skyline cars
	vector<int> C_idx;
	for(int i = 0; i < P->numberOfPoints; i++)
		C_idx.push_back(i);

	double time;

	// the initial exteme vector sets V = {−ei | i ∈ [1, d], ei [i] = 1 and ei [j] = 0 if i , j}.
	vector<point_t*> ext_vec;
	for (int i = 0; i < dim; i++)
	{
		point_t* e = alloc_point(dim);
		for (int j = 0; j < dim; j++)
		{
			if (i == j)
				e->coord[j] = -1;
			else
				e->coord[j] = 0;
		}
		ext_vec.push_back(e);
	}

	int current_best_idx = -1;
	int last_best = -1;
	vector<int> frame;

	// get the index of the "current best" point
	//if(cmp_option != RANDOM)
	current_best_idx = get_current_best_pt(P, C_idx, ext_vec);
	
	// if not skyline
	//sql_pruning(P, C_idx, ext_vec);

	// Qcount - the number of querstions asked
	// Csize - the size of the current candidate set

	Qcount = 0;
	double rr = 1;

	int max_i;
	double max;

	// interactively reduce the candidate set and shrink the candidate utility range
	while (Qcount <  maxRound)  // while none of the stopping conditiong is true
	{
		Qcount++;
		sort(C_idx.begin(), C_idx.end()); // prevent select two different points after different skyline algorithms

		// generate s cars for selection in a round
		vector<int> S_idx = generate_S(P, C_idx, s, ext_vec, current_best_idx, last_best, frame, cmp_option, stop_option, prune_option, dom_option, epsilon, delta, repeats);
		
		vector<point_t*> S_real(S_idx.size());
		for(int i = 0; i < S_idx.size(); ++i)
		  S_real[i] = P->points[C_idx[S_idx[i]]];

		//cout << "** computed S of size " << S_idx.size() << endl;

		// if we have only one skyline point left, we can prune down to its optimal and quit
		if (S_idx.size() == 1)
		{
		  double max_value = dot_prod(u, P->points[C_idx[S_idx[0]]]);
		  vector<int> I_idx;
		  for(int i = 0; i < C_idx.size(); ++i)
		    if (dot_prod(u, P->points[C_idx[i]]) * (1 + epsilon) >= max_value)
		      I_idx.push_back(C_idx[i]);
		  C_idx = I_idx;
		  break; 
		}

		// find the user's favorite point in S_real
		max = -1;
		for(int i = 0; i < S_real.size(); i++)
		{
		  double v = dot_prod(u, S_real[i]);
		  if(v > max)
		    max = v;
		}

		// randomly select one of the delta-indistinguishable ones
		vector<int> options;
		for(int i = 0; i < S_real.size(); i++)
		{
		  double v = dot_prod(u, S_real[i]);
		  if(v * (1 + delta) >= max)
		    options.push_back(i);
		}
		max_i = options[rand() % options.size()];

		//cout << "** got user's fave\n";
		//cout << max << " " << max_i << endl;
		

		// get the better car among those from the user
		last_best = current_best_idx;
		current_best_idx = C_idx[S_idx[max_i]];
		
		//cout << "** swapped last and current\n";

		// update the extreme vectors based on the user feedback
		update_ext_vec(P, S_real, C_idx, max_i, s, ext_vec, current_best_idx, last_best, frame, cmp_option, delta);

		//cout << "** updated ext of size " << ext_vec.size() << "\n";

		if(C_idx.size()==1 ) // || global_best_idx == current_best_idx
			break;

		//update candidate set
		if(prune_option == SQL)
		  sql_pruning(P, C_idx, ext_vec, rr, stop_option, dom_option, epsilon);
		else
		  rtree_pruning(P, C_idx, ext_vec, rr, stop_option, dom_option, epsilon);

		//printf("Round %d, pruned down to %d tuples.\n", Qcount, C_idx.size());

	}

	
	double max_value = 0;
	for(int i = 0; i < P->numberOfPoints; i++)
	  {
	    double value = dot_prod(u, P->points[i]);
	    if(value > max_value)
		max_value = value;
	  }


	/*
	int inI = 0;
	double alpha = 0.0;
	for(int i = 0; i < C_idx.size(); i++)
	  {
	    double value = dot_prod(u, P->points[C_idx[i]]);
	    if(value * (1 + epsilon) > max_value)
	      inI++;

	    if (max_value - value * (1 + epsilon) > alpha)
	      alpha = max_value - value * (1 + epsilon);

	  }
	printf("Found %d in I and %d false positives and alpha was %lf.\n", inI, C_idx.size() - inI, alpha);
	*/

	int inI = 0;
	double alpha = 0.0;
	double avg_effective_epsilon = 0.0, max_effective_epsilon = 0.0;
	for(int i = 0; i < C_idx.size(); i++)
	{
	  double value = dot_prod(u, P->points[C_idx[i]]);
	  if(value * (1 + epsilon) > max_value)
	    inI++;
	  else
	  {
	    avg_effective_epsilon += max_value/value - 1.0;
	    if (max_value/value - 1.0 > max_effective_epsilon)
	      max_effective_epsilon = max_value/value - 1.0;

	    if (max_value - value * (1 + epsilon) > alpha)
	      alpha = max_value - value * (1 + epsilon);
	  }
	}
	if (C_idx.size() - inI > 0)
	  avg_effective_epsilon /= C_idx.size() - inI;
	printf("Found %d in I; %d false positives; alpha was %lf; avg effective epsilon was %lf.; max effective epsilon was %lf\n", inI, C_idx.size() - inI, alpha, avg_effective_epsilon, max_effective_epsilon);

	// get the final result 
	point_t* result = P->points[get_current_best_pt(P, C_idx, ext_vec)];
	Csize = C_idx.size();

	for (int i = 0; i < ext_vec.size(); i++)
	  release_point(ext_vec[i]);

	return alpha;
}


// the main interactive algorithm for fake points
// P: the input dataset (assumed (1+epsilon)-skyline)
// u: the unknown utility vector
// s: the question size
// epsilon: the indistinguishability threshold
// maxRound: the maximum number of rounds of interacitons
// Qcount: the number of question asked
// Csize: the size the candidate set when finished
double max_utility_fake(point_set_t* P, point_t* u, int s,  double epsilon, double delta, int maxRound, int &Qcount, int &Csize)
{
  int dim = P->points[0]->dim;

  // the indexes of the candidate set
  // initially, it is all the skyline cars
  vector<int> C_idx;
  for(int i = 0; i < P->numberOfPoints; i++)
    C_idx.push_back(i);  
  
  // if we have fewer than (d-1)/(s-1) rounds we don't have enough questions to discover i^*,
  // so we just return the current set
  if (maxRound <= (dim - 1)/(s - 1))
  {
    double max_value = 0;
    for(int i = 0; i < P->numberOfPoints; i++)
    {
      double value = dot_prod(u, P->points[i]);
      if(value > max_value)
	max_value = value;
    }
    double alpha = 0.0;
    for(int i = 0; i < C_idx.size(); i++)
    {
      double value = dot_prod(u, P->points[C_idx[i]]);
      if (max_value - value * (1 + epsilon) > alpha)
	alpha = max_value - value * (1 + epsilon);
    }
    return alpha;
  }

  if ((dim - 1) % (s - 1) == 0)
    Qcount = (dim - 1) / (s - 1);
  else
    Qcount = (dim - 1) / (s - 1) + 1; // round up

  
  // go through the dimensions (s-1) at a time to find the max u_i,
  // making sure to randomly select an option if there are any points
  // that are delta-indistinguishable from the best so far
  int istar = 0;
  double max_ui = u->coord[0];
  for(int i = 1; i < dim; i += s - 1)
  {
    double best = max_ui;
    for(int j = i; j < dim && j < i + s - 1; ++j)
      if (u->coord[j] >= best)
	best = u->coord[j];

    vector<int> options;
    if (u->coord[istar] * (1 + delta) >= best)    // if the best so far is big enough
      options.push_back(istar);                   // it is a contender
    for(int j = i; j < dim && j < i + s - 1; ++j)
      if (u->coord[j] * (1 + delta) >= best)
	options.push_back(j);
    
    istar = options[rand() % options.size()];
    max_ui = u->coord[istar];
  }
	
  double upperbound = (dim - 1) % (s - 1) == 0 ? pow(1 + delta, (dim - 1)/(s - 1)) : pow(1 + delta, (dim - 1)/(s - 1) + 1);
  vector<double> L(dim, 0.0), H(dim, upperbound);
  L[istar] = 1;
  H[istar] = 1;

  int i = 0;
  while(Qcount < maxRound)
  {
    if (i == istar)
      i = (i + 1) % dim;

    // compute chi values
    vector<double> chi(s + 1);
    for (int j = 0; j < s + 1; ++j)
      chi[j] = L[i] + j * (H[i] - L[i]) / s;

    // find the best of the s points p1,...,ps
    int c = 0;
    double best_val = 0;
    for(int j = 1; j <= s; ++j)
    {
      double sum = 0.0;
      for(int k = j; k <= s - 1; ++k)
	sum += chi[k] / s;

      // sum up the utility for the attributes i and istar as the points are zero elsewhere
      double val = (u->coord[i] * j) / s + u->coord[istar] * sum;
      
      if (val > best_val)
	{
	  c = j;
	  best_val = val;
	}
    }

    // find all indistinguishable options
    vector<int> options;
    for(int j = 1; j <= s; ++j)
    {
      double sum = 0.0;
      for(int k = j; k <= s - 1; ++k)
	sum += chi[k] / s;

      double val = (u->coord[i] * j) / s + u->coord[istar] * sum;
      if ((1 + delta) * val >= best_val)
	options.push_back(j);
    }
      
    // choose an option randomly from indistinguishable options
    c = options[rand() % options.size()];

    double sum = 0.0;
    for(int j = c; j < s; ++j)
      sum += chi[j];

    // update L_i and H_i based on this c
    L[i] = max(L[i], (chi[c - 1] - delta * sum)/(1 + c * delta));     //max(L[i], chi[c - 1] - delta * (2 * s - 1));
    H[i] = min(H[i], (chi[c] + delta * sum)/(1 - c * delta));         //min(H[i], chi[c]     + delta * (2 * s -  1));
    
    i = (i + 1) % dim;
    Qcount += 1;

    //cout << Qcount << endl;
    //for (int j = 0; j < dim; ++j)
    //  printf("%d: %lf <= %lf <= %lf\n", j, L[j], u->coord[j]/max_ui, H[j]);
    //cout << endl;

  }

  // debug at the approximation of u
  //for (int j = 0; j < dim; ++j)
  //  printf("%d: %lf <= %lf <= %lf\n", j, L[j], u->coord[j]/max_ui, H[j]);

  // find the highest value from the low-end of the user utilities
  double highest = 0.0;
  for (int j = 0; j < P->numberOfPoints; ++j)
  {
    double dot = 0.0;
    for(int k = 0; k < dim; ++k)
      dot += P->points[j]->coord[k] * L[k];
    if (dot > highest)
      highest = dot;
  }

  // prune all the points that have their high-end utility (1+epsilon) dominated
  C_idx.clear();
  for (int j = 0; j < P->numberOfPoints; ++j)
  {
    double dot = 0.0;
    for(int k = 0; k < dim; ++k)
      dot += P->points[j]->coord[k] * H[k];
    if (dot * (1 + epsilon) >= highest)
      C_idx.push_back(j);
  }

  // Find out how well this did:
  double max_value = 0;
  for(int i = 0; i < P->numberOfPoints; i++)
  {
    double value = dot_prod(u, P->points[i]);
    if(value > max_value)
      max_value = value;
  }

  int inI = 0;
  double alpha = 0.0;
  double avg_effective_epsilon = 0.0, max_effective_epsilon = 0.0;
  for(int i = 0; i < C_idx.size(); i++)
    {
      double value = dot_prod(u, P->points[C_idx[i]]);
      if(value * (1 + epsilon) > max_value)
	inI++;
      else
	{
	  avg_effective_epsilon += max_value/value - 1.0;
	  if (max_value/value - 1.0 > max_effective_epsilon)
	    max_effective_epsilon = max_value/value - 1.0;

	  if (max_value - value * (1 + epsilon) > alpha)
	    alpha = max_value - value * (1 + epsilon);
	}
    }
  if (C_idx.size() - inI > 0)
    avg_effective_epsilon /= C_idx.size() - inI;
  printf("Found %d in I; %d false positives; alpha was %lf; avg effective epsilon was %lf; max effective epsilon was %lf.\n", inI, C_idx.size() - inI, alpha, avg_effective_epsilon, max_effective_epsilon);
  Csize = C_idx.size();

  return alpha;
}

// the random algorithm for fake points
// P: the input dataset (assumed (1+epsilon)-skyline)
// u: the unknown utility vector
// s: the question size
// epsilon: the indistinguishability threshold
// maxRound: the maximum number of rounds of interacitons
// Qcount: the number of question asked
// Csize: the size of the candidate set when finished
double random_fake(point_set_t* P, point_t* u, int s,  double epsilon, double delta, int maxRound, int &Qcount, int &Csize)
{
  int dim = P->points[0]->dim;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option = RANDOM;
  int current_best_idx = -1;
  int last_best = -1;
  vector<int> frame;
  vector<point_t*> ext_pts;
  vector<point_t*> hyperplanes;
  hyperplane_t* hp = NULL;

  // the indexes of the candidate set
  // initially, it is all the skyline cars
  vector<int> C_idx;
  for(int i = 0; i < P->numberOfPoints; i++)
    C_idx.push_back(i);

  // the initial exteme vector sets V = {?ei | i ? [1, d], ei [i] = 1 and ei [j] = 0 if i , j}.
  vector<point_t*> ext_vec;
  for (int i = 0; i < dim; i++)
    {
      point_t* e = alloc_point(dim);
      for (int j = 0; j < dim; j++)
	{
	  if (i == j)
	    e->coord[j] = -1;
	  else
	    e->coord[j] = 0;
	}
      ext_vec.push_back(e);
    }

  if(dom_option == HYPER_PLANE)
    ext_pts = get_extreme_pts(ext_vec); // in Hyperplane Pruning, we need the set of extreme points of R
  else
    {
      // in Conical Pruning, we need bounding hyperplanes for the conical hull
      get_hyperplanes(ext_vec, hp, hyperplanes);
      if(stop_option != NO_BOUND) // if an upper bound on the regret ratio is needed, we need the set of extreme points of R
	ext_pts = get_extreme_pts(ext_vec);
    }

  // get the index of the "current best" point
  //if(cmp_option != RANDOM)
  //current_best_idx = get_current_best_pt(P, C_idx, ext_vec);

  double rr = 1;

  for(int round = 0; round < maxRound; ++round)
  {
    // generate s random points
    vector<point_t*> S(s);
    for(int i = 0; i < s; ++i)
      S[i] = rand_point(dim);

    // see what the user picks among delta-indistinguishable options
    int best = 0;
    double best_value = 0.0;
    for(int i = 0; i < s; ++i)
    {
      double value = dot_prod(u, S[i]);
      if (value > best_value)
      {
	best = i;
	best_value = value;
      }
    }
    vector<int> options;    // list of optimally delta-indistinguishable points among the s
    for(int i = 0; i < s; ++i)
    {
      double value = dot_prod(u, S[i]);
      if (value * (1 + delta) >= best_value)
	options.push_back(i);
    }
    int max_i = options[rand() % options.size()];

    // update the extreme vectors based on the user feedback
    update_ext_vec(P, S, C_idx, max_i, s, ext_vec, current_best_idx, last_best, frame, cmp_option, delta);

    Qcount = round;

    if(C_idx.size()==1 ) // || global_best_idx == current_best_idx
      break;

    //update candidate set
    if(prune_option == SQL)
      sql_pruning(P, C_idx, ext_vec, rr, stop_option, dom_option, epsilon);
    else
      rtree_pruning(P, C_idx, ext_vec, rr, stop_option, dom_option, epsilon);

    // release memory for random points
    for(int i = 0; i < s; ++i)
      release_point(S[i]);
  }

  // Find out how well this did:
  double max_value = 0;
  for(int i = 0; i < P->numberOfPoints; i++)
    {
      double value = dot_prod(u, P->points[i]);
      if(value > max_value)
	max_value = value;
    }

  int inI = 0;
  double alpha = 0.0;
  double avg_effective_epsilon = 0.0, max_effective_epsilon = 0.0;
  for(int i = 0; i < C_idx.size(); i++)
    {
      double value = dot_prod(u, P->points[C_idx[i]]);
      if(value * (1 + epsilon) > max_value)
	inI++;
      else
	{
	  if (max_value/value - 1.0 > max_effective_epsilon)
            max_effective_epsilon = max_value/value - 1.0;

	  avg_effective_epsilon += max_value/value - 1.0;

	  if (max_value - value * (1 + epsilon) > alpha)
	    alpha = max_value - value * (1 + epsilon);
	}
    }
  if (C_idx.size() - inI > 0)
    avg_effective_epsilon /= C_idx.size() - inI;
  printf("Found %d in I; %d false positives; alpha was %lf; avg effective epsilon was %lf; max effective epsilon was %lf.\n", inI, C_idx.size() - inI, alpha, avg_effective_epsilon, max_effective_epsilon);
  
  Csize = C_idx.size();

  for(int i = 0; i < ext_pts.size(); i++)
    release_point(ext_pts[i]);
  
  for(int i = 0; i < ext_vec.size(); i++)
    release_point(ext_vec[i]);

  return alpha;
}
