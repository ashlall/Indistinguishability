#include "data_utility.h"
#include "data_struct.h"
#include "read_write.h"
#include "maxUtility.h"
#include "medianhull.h"
#include <vector>
#include <ctime>
using namespace std;

void run_test(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q);
void run_vary_s(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_vary_q(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_vary_epsilon(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_vary_delta(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_vary_epsilon_delta(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_vary_T(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q);
void run_fix_n(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_false_positives(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);
void run_time(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake);

int main(int argc, char *argv[])
{
  // parameters
  int s = 2;
  double epsilon = 0.01, delta;
  int q = 20;                // maximum number of rounds to run for
  int Qcount, Csize;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha = 1;                   // factor by which we are willing to overestimate the size of the output set
 
  char* filename = "nba.txt";   // TODO: Update type in case too long
  char type;
  bool real, fake;

  if (argc < 9)
  {
    cout << "Usage: run type[S/Q/T/E/D] filename epsilon delta s q [R]eal/[F]ake/[B]oth seed" << endl;
    return 0;
  }

  type = argv[1][0];

  filename = argv[2];

  epsilon = stod(argv[3]);

  delta = stod(argv[4]);

  s = stoi(argv[5]);

  q = stoi(argv[6]);

  real = argv[7][0] == 'R' || argv[7][0] == 'r' || argv[7][0] == 'B' || argv[7][0] == 'b';
  fake = argv[7][0] == 'F' || argv[7][0] == 'f' || argv[7][0] == 'B' || argv[7][0] == 'b';

  // use the random seed as a parameter to get repeatable results
  srand(atoi(argv[8]));

  printf("file = %s\nepsilon = %lf, delta = %lf, s = %d, q = %d\n", filename, epsilon, delta, s, q);

  // read the dataset
  point_set_t* P = read_points(filename);
  cout << "n = " << P->numberOfPoints << endl;

  point_set_t* skyline = c_skyline_point(1.0, P);
  cout << "skyline size = " << skyline->numberOfPoints << endl;

  point_set_t* cskyline = c_skyline_point(1.0 + epsilon, P);
  cout << "c-skyline size = " << cskyline->numberOfPoints << endl;

  int dim = P->points[0]->dim;

  switch (type)
  {
  case 'S':   // vary s, the number of questions per round
    run_vary_s(P, cskyline, epsilon, delta, s, q, real, fake);
      
    break;
  case 'Q':   // vary q, the number of total questions
    run_vary_q(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'N':  // fix the number of total points
    run_fix_n(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'E':
    run_vary_epsilon(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'D':
    run_vary_delta(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'B':
    run_vary_epsilon_delta(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'F':
    run_false_positives(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 't':
    run_time(P, cskyline, epsilon, delta, s, q, real, fake);
    break;

  case 'T':  // vary T, the number of repeats for MinR and MinC algorithms
    run_vary_T(P, cskyline, epsilon, delta, s, q);
    break;

  default:  // for testing purposes
    run_test(P, cskyline, epsilon, delta, s, q);
    break;
  }

  release_point_set(cskyline, false);
  release_point_set(skyline, false);
  release_point_set(P, true);
  
  return 0;
}
  
void run_test(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha = 1;                   // factor by which we are willing to overestimate the size of the output set
  int reps = 5;

  // generate the utility vector
  point_t* u = alloc_point(dim);
  for (int i = 0; i < dim; i++)
    u->coord[i] = ((double)rand()) / RAND_MAX;

  // look for the ground truth maximum utility point
  int maxIdx = 0;
  double maxValue = 0;
  for(int i = 0; i < cskyline->numberOfPoints; i++)
    {
      double value = dot_prod(u, cskyline->points[i]);
      if(value > maxValue)
	{
	  maxValue = value;
	  maxIdx = i;
	}
    }

  // count the number of points that are within epsilon of the optimal
  int Isize = 0;
  for(int i = 0; i < P->numberOfPoints; i++)
    {
      double value = dot_prod(u, P->points[i]);
      if(value * (1 + epsilon) > maxValue)
	Isize++;
    }


  double alpha_SU = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
  double alpha_RF = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
  double alpha_MC = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
  double alpha_MR = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);

  printf("%10s : %lf\n", "SqueezeU", alpha_SU);
  printf("%10s : %lf\n", "RandomFake", alpha_RF);
  printf("%10s : %lf\n", "MinC", alpha_MC);
  printf("%10s : %lf\n", "MinR", alpha_MR);

  release_point(u);
}

void run_vary_s(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10; // number of times to repeat experiment
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  int MIN_S = 2, MAX_S = 10; // MAX_S < 100
  int reps = 50; // number of times needed my MinR and MinC
  
  for(s = MIN_S; s <= MAX_S; s += 2)
  {
    cout << "s = " << s << endl;
    avg_alpha[SQUEEZEU][s] = 0.0;
    avg_alpha[RANDFAKE][s] = 0.0;
    avg_alpha[MIN_C][s] = 0.0;
    avg_alpha[MIN_R][s] = 0.0;

    for(int repeat = 0; repeat < REPEATS; ++repeat)
    {
      // generate a random utility function and make the max equal to 1
      point_t* u = alloc_point(dim);
      double max_u = 0;
      for (int i = 0; i < dim; i++)
      {
	u->coord[i] = ((double)rand());
	if (u->coord[i] > max_u) max_u = u->coord[i];
      }
      for (int i = 0; i < dim; i++)
	u->coord[i] /= max_u;

      // look for the ground truth maximum utility point
      int maxIdx = 0;
      double maxValue = 0;
      for(int i = 0; i < cskyline->numberOfPoints; i++)
      {
	double value = dot_prod(u, cskyline->points[i]);
	if(value > maxValue)
	{
	  maxValue = value;
	  maxIdx = i;
	}
      }
      
      // count the number of points that are within epsilon of the optimal
      int Isize = 0;
      for(int i = 0; i < P->numberOfPoints; i++)
      {
	  double value = dot_prod(u, P->points[i]);
	  if(value * (1 + epsilon) > maxValue)
	    Isize++;
      }

      if (fake)
      {	
	alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	avg_alpha[SQUEEZEU][s] += alpha/REPEATS;

	alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
        avg_alpha[RANDFAKE][s] += alpha/REPEATS;
      }

      if (real)
      {
	alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	avg_alpha[MIN_C][s] += alpha/REPEATS;

	alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
        avg_alpha[MIN_R][s] += alpha/REPEATS;
      }

    }
  }

  char filename[200];
  sprintf(filename, "output/vary-s-%d-%d-e%lf-d%lf-q%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, q, time(NULL));
  FILE* fp = fopen(filename, "w");

  for(s = MIN_S; s <= MAX_S; s += 2)
  {
    printf("%d\t%lf\t%lf\t%lf\t%lf\n", s, avg_alpha[SQUEEZEU][s], avg_alpha[RANDFAKE][s], avg_alpha[MIN_C][s], avg_alpha[MIN_R][s]);
    fprintf(fp, "%d\t%lf\t%lf\t%lf\t%lf\n", s, avg_alpha[SQUEEZEU][s], avg_alpha[RANDFAKE][s], avg_alpha[MIN_C][s], avg_alpha[MIN_R][s]);
  }
  fclose(fp);

}


void run_vary_q(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10; // number of times to repeat experiment
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  int MIN_Q = 5, MAX_Q = 15; // MAX < 100
  int reps = 50; // number of times needed my MinR and MinC

  for(q = MIN_Q; q <= MAX_Q; q += 2)
    {
      cout << "q = " << q << endl;
      avg_alpha[SQUEEZEU][q] = 0.0;
      avg_alpha[RANDFAKE][q] = 0.0;
      avg_alpha[MIN_C][q] = 0.0;
      avg_alpha[MIN_R][q] = 0.0;

      for(int repeat = 0; repeat < REPEATS; ++repeat)
	{
	  // generate a random utility function and make the max equal to 1
	  point_t* u = alloc_point(dim);
	  double max_u = 0;
	  for (int i = 0; i < dim; i++)
	    {
	      u->coord[i] = ((double)rand());
	      if (u->coord[i] > max_u) max_u = u->coord[i];
	    }
	  for (int i = 0; i < dim; i++)
	    u->coord[i] /= max_u;

	  if (fake)
	    {
	      alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	      avg_alpha[SQUEEZEU][q] += alpha/REPEATS;

	      alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	      avg_alpha[RANDFAKE][q] += alpha/REPEATS;
	    }

	  if (real)
	    {
	      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	      avg_alpha[MIN_C][q] += alpha/REPEATS;

	      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	      avg_alpha[MIN_R][q] += alpha/REPEATS;
	    }

	}
    }

  char filename[200];
  sprintf(filename, "output/vary-q-%d-%d-e%lf-d%lf-s%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, s, time(NULL));
  FILE* fp = fopen(filename, "w");

  for(q = MIN_Q; q <= MAX_Q; q += 2)
    {
      printf("%d\t%lf\t%lf\t%lf\t%lf\n", q, avg_alpha[SQUEEZEU][q], avg_alpha[RANDFAKE][q], avg_alpha[MIN_C][q], avg_alpha[MIN_R][q]);
      fprintf(fp, "%d\t%lf\t%lf\t%lf\t%lf\n", q, avg_alpha[SQUEEZEU][q], avg_alpha[RANDFAKE][q], avg_alpha[MIN_C][q], avg_alpha[MIN_R][q]);
    }
  fclose(fp);
}

void run_fix_n(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10; // number of times to repeat experiment
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  int MIN_S = 2, MAX_S = 6; // MAX < 100
  int reps = 50; // number of times needed my MinR and MinC
  int total_points = 24;

  for(s = MIN_S; s <= MAX_S; s += 1)
    {
      q = total_points / s;
      cout << "s = " << s << endl;
      cout << "q = " << q << endl;
      avg_alpha[SQUEEZEU][s] = 0.0;
      avg_alpha[RANDFAKE][s] = 0.0;
      avg_alpha[MIN_C][s] = 0.0;
      avg_alpha[MIN_R][s] = 0.0;

      for(int repeat = 0; repeat < REPEATS; ++repeat)
        {
          // generate a random utility function and make the max equal to 1
          point_t* u = alloc_point(dim);
          double max_u = 0;
          for (int i = 0; i < dim; i++)
            {
              u->coord[i] = ((double)rand());
              if (u->coord[i] > max_u) max_u = u->coord[i];
            }
          for (int i = 0; i < dim; i++)
            u->coord[i] /= max_u;

          if (fake)
            {
              alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
              avg_alpha[SQUEEZEU][s] += alpha/REPEATS;

              alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
              avg_alpha[RANDFAKE][s] += alpha/REPEATS;
            }

	  if (real)
            {
              alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
              avg_alpha[MIN_C][s] += alpha/REPEATS;

              alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
              avg_alpha[MIN_R][s] += alpha/REPEATS;
            }

        }
    }

  char filename[200];
  sprintf(filename, "output/fix-n-%d-%d-e%lf-d%lf-n%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, total_points, time(NULL));
  FILE* fp = fopen(filename, "w");

  for(s = MIN_S; s <= MAX_S; s += 1)
    {
      printf("%d\t%lf\t%lf\t%lf\t%lf\n", s, avg_alpha[SQUEEZEU][s], avg_alpha[RANDFAKE][s], avg_alpha[MIN_C][s], avg_alpha[MIN_R][s]);
      fprintf(fp, "%d\t%lf\t%lf\t%lf\t%lf\n", s, avg_alpha[SQUEEZEU][s], avg_alpha[RANDFAKE][s], avg_alpha[MIN_C][s], avg_alpha[MIN_R][s]);
    }
  fclose(fp);
}

void run_vary_epsilon(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10, reps = 50;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  double MIN_E = 0.001, FACTOR_E = 10;

  epsilon = MIN_E;
  for(int i = 0; i < 3; ++i)
  {
    cout << "epsilon = " << epsilon << endl;
    avg_alpha[SQUEEZEU][i] = 0.0;
    avg_alpha[RANDFAKE][i] = 0.0;
    avg_alpha[MIN_C][i] = 0.0;
    avg_alpha[MIN_R][i] = 0.0;

    for(int repeat = 0; repeat < REPEATS; ++repeat)
      {
	// generate a random utility function and make the max equal to 1
	point_t* u = alloc_point(dim);
	double max_u = 0;
	for (int i = 0; i < dim; i++)
	  {
	    u->coord[i] = ((double)rand());
	    if (u->coord[i] > max_u) max_u = u->coord[i];
	  }
	for (int i = 0; i < dim; i++)
	  u->coord[i] /= max_u;

	if (fake)
	  {
	    alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	    avg_alpha[SQUEEZEU][i] += alpha/REPEATS;

	    alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	    avg_alpha[RANDFAKE][i] += alpha/REPEATS;
	  }

	if (real)
	  {
	    alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	    avg_alpha[MIN_C][i] += alpha/REPEATS;

	    alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	    avg_alpha[MIN_R][i] += alpha/REPEATS;
	  }

      }

    epsilon *= FACTOR_E;
  }

  char filename[200];
  sprintf(filename, "output/vary-e-%d-%d-d%lf-q%d-s%d-%d.dat", dim, P->numberOfPoints, delta, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");

  epsilon = MIN_E;
  for(int i = 0; i < 3; ++i)
    {
      printf("%lf\t%lf\t%lf\t%lf\t%lf\n", epsilon, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\n", epsilon, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      epsilon *= FACTOR_E;
    }
  fclose(fp);
}

void run_vary_delta(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10, reps = 50;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  double MIN_D = 0.001, FACTOR_D = 10;

  delta = MIN_D;
  for(int i = 0; i < 3; ++i)
    {
      cout << "delta = " << delta << endl;
      avg_alpha[SQUEEZEU][i] = 0.0;
      avg_alpha[RANDFAKE][i] = 0.0;
      avg_alpha[MIN_C][i] = 0.0;
      avg_alpha[MIN_R][i] = 0.0;

      for(int repeat = 0; repeat < REPEATS; ++repeat)
	{
	  // generate a random utility function and make the max equal to 1
	  point_t* u = alloc_point(dim);
	  double max_u = 0;
	  for (int i = 0; i < dim; i++)
	    {
	      u->coord[i] = ((double)rand());
	      if (u->coord[i] > max_u) max_u = u->coord[i];
	    }
	  for (int i = 0; i < dim; i++)
	    u->coord[i] /= max_u;

	  if (fake)
	    {
	      alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	      avg_alpha[SQUEEZEU][i] += alpha/REPEATS;

	      alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	      avg_alpha[RANDFAKE][i] += alpha/REPEATS;
	    }

	  if (real)
	    {
	      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	      avg_alpha[MIN_C][i] += alpha/REPEATS;

	      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	      avg_alpha[MIN_R][i] += alpha/REPEATS;
	    }
	}
      
      delta *= FACTOR_D;
    }

  char filename[200];
  sprintf(filename, "output/vary-d-%d-%d-e%lf-q%d-s%d-%d.dat", dim, P->numberOfPoints, epsilon, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");

  delta = MIN_D;
  for(int i = 0; i < 3; ++i)
    {
      printf("%lf\t%lf\t%lf\t%lf\t%lf\n", delta, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\n", delta, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      delta *= FACTOR_D;
    }
  fclose(fp);
}

void run_vary_epsilon_delta(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 10, reps = 50;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  double MIN_D = 0.001, FACTOR_D = 10;
  
  delta = MIN_D;
  for(int i = 0; i < 3; ++i)
  {
    cout << "epsilon = delta = " << delta << endl;
    epsilon = delta;
    avg_alpha[SQUEEZEU][i] = 0.0;
    avg_alpha[RANDFAKE][i] = 0.0;
    avg_alpha[MIN_C][i] = 0.0;
    avg_alpha[MIN_R][i] = 0.0;
    
    for(int repeat = 0; repeat < REPEATS; ++repeat)
      {
	// generate a random utility function and make the max equal to 1
	point_t* u = alloc_point(dim);
	double max_u = 0;
	for (int i = 0; i < dim; i++)
	  {
	    u->coord[i] = ((double)rand());
	    if (u->coord[i] > max_u) max_u = u->coord[i];
	  }
	for (int i = 0; i < dim; i++)
	  u->coord[i] /= max_u;
	
	if (fake)
	  {
	    alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	    avg_alpha[SQUEEZEU][i] += alpha/REPEATS;
	    
	    alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	    avg_alpha[RANDFAKE][i] += alpha/REPEATS;
	  }
	
	if (real)
	  {
	    alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	    avg_alpha[MIN_C][i] += alpha/REPEATS;
	    
	    alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	    avg_alpha[MIN_R][i] += alpha/REPEATS;
	  }
      }
    
    delta *= FACTOR_D;
  }
  
  char filename[200];
  sprintf(filename, "output/vary-ed-%d-%d-q%d-s%d-%d.dat", dim, P->numberOfPoints, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");
  
  delta = MIN_D;
  for(int i = 0; i < 3; ++i)
    {
      printf("%lf\t%lf\t%lf\t%lf\t%lf\n", delta, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\n", delta, avg_alpha[SQUEEZEU][i], avg_alpha[RANDFAKE][i], avg_alpha[MIN_C][i], avg_alpha[MIN_R][i]);
      delta *= FACTOR_D;
    }
    fclose(fp);
}

void run_false_positives(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, C_size[5];
  int REPEATS = 10, reps = 50;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3, REAL = 4;
  
  for(int i = 0; i < 5; ++i) 
    C_size[i] = 0.0;
  
  for(int repeat = 0; repeat < REPEATS; ++repeat)
  {
    // generate a random utility function and make the max equal to 1
    point_t* u = alloc_point(dim);
    double max_u = 0;
    for (int i = 0; i < dim; i++)
      {
	u->coord[i] = ((double)rand());
	if (u->coord[i] > max_u) max_u = u->coord[i];
      }
    for (int i = 0; i < dim; i++)
      u->coord[i] /= max_u;

    if (fake)
      {
	alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	C_size[SQUEEZEU] += ((double)Csize)/REPEATS;

	alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	C_size[RANDFAKE] += ((double)Csize)/REPEATS;
      }

    if (real)
      {
	alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	C_size[MIN_C] += ((double)Csize)/REPEATS;

	alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	C_size[MIN_R] += ((double)Csize)/REPEATS;
      } 


    // look for the ground truth maximum utility point
    int maxIdx = 0;
    double maxValue = 0;
    for(int i = 0; i < cskyline->numberOfPoints; i++)
      {
	double value = dot_prod(u, cskyline->points[i]);
	if(value > maxValue)
	  {
	    maxValue = value;
	    maxIdx = i;
	  }
      }

    // count the number of points that are within epsilon of the optimal
    int Isize = 0;
    for(int i = 0; i < P->numberOfPoints; i++)
      {
	double value = dot_prod(u, P->points[i]);
	if(value * (1 + epsilon) > maxValue)
	  Isize++;
      }

    C_size[REAL] += ((double)Isize)/REPEATS;

  }

  char filename[200];
  sprintf(filename, "output/fp-%d-%d-e%lf-d%lf-q%d-s%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");
  printf("%lf\t%lf\t%lf\t%lf\t%lf\n", C_size[SQUEEZEU], C_size[RANDFAKE], C_size[MIN_C], C_size[MIN_R], C_size[REAL]);
  fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\n", C_size[SQUEEZEU], C_size[RANDFAKE], C_size[MIN_C], C_size[MIN_R], C_size[REAL]);
  fclose(fp);
}


void run_time(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q, bool real, bool fake)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, t[5];
  int REPEATS = 10, reps = 50;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  time_t start, finish;
  double elapsed_time;

  for(int i = 0; i < 5; ++i)
    t[i] = 0.0;

  for(int repeat = 0; repeat < REPEATS; ++repeat)
    {
      // generate a random utility function and make the max equal to 1
      point_t* u = alloc_point(dim);
      double max_u = 0;
      for (int i = 0; i < dim; i++)
	{
	  u->coord[i] = ((double)rand());
	  if (u->coord[i] > max_u) max_u = u->coord[i];
	}
      for (int i = 0; i < dim; i++)
	u->coord[i] /= max_u;

      if (fake)
	{
	  time(&start);
	  alpha = max_utility_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	  time(&finish);
	  elapsed_time = difftime(finish, start);
	  t[SQUEEZEU] += elapsed_time/REPEATS;

	  time(&start);
	  alpha = random_fake(cskyline, u, s, epsilon, delta, q, Qcount, Csize);
	  time(&finish);
          elapsed_time = difftime(finish, start);
	  t[RANDFAKE] += elapsed_time/REPEATS;
	}

      if (real)
	{
	  time(&start);
	  alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, reps);
	  time(&finish);
          elapsed_time = difftime(finish, start);
	  t[MIN_C] += elapsed_time/REPEATS;

	  time(&start);
	  alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, reps);
	  time(&finish);
          elapsed_time = difftime(finish, start);
	  t[MIN_R] += elapsed_time/REPEATS;
	}
    }

  char filename[200];
  sprintf(filename, "output/time-%d-%d-e%lf-d%lf-q%d-s%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");
  printf("%lf\t%lf\t%lf\t%lf\n", t[SQUEEZEU], t[RANDFAKE], t[MIN_C], t[MIN_R]);
  fprintf(fp, "%lf\t%lf\t%lf\t%lf\n", t[SQUEEZEU], t[RANDFAKE], t[MIN_C], t[MIN_R]);
  fclose(fp);
}


void run_vary_T(point_set_t* P, point_set_t* cskyline, double epsilon, double delta, int s, int q)
{
  int dim = P->points[0]->dim;
  int Qcount, Csize;
  point_t* p;
  int prune_option = RTREE;
  int dom_option = HYPER_PLANE;
  int stop_option = EXACT_BOUND;
  int cmp_option;
  double alpha, avg_alpha[4][100];
  int REPEATS = 5;
  int SQUEEZEU = 0, RANDFAKE = 1, MIN_C = 2, MIN_R = 3;
  int MIN_T = 10, MAX_T = 100; // MAX_S < 100

  for(int T = MIN_T; T <= MAX_T; T += 20)
  {
    cout << "T = " << T << endl;
    avg_alpha[MIN_C][T] = 0.0;
    avg_alpha[MIN_R][T] = 0.0;

    for(int repeat = 0; repeat < REPEATS; ++repeat)
    {
      // generate a random utility function and make the max equal to 1
      point_t* u = alloc_point(dim);
      double max_u = 0;
      for (int i = 0; i < dim; i++)
	{
	  u->coord[i] = ((double)rand());
	  if (u->coord[i] > max_u) max_u = u->coord[i];
	}
      for (int i = 0; i < dim; i++)
	u->coord[i] /= max_u;
      
      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINC, stop_option, prune_option, dom_option, T);
      avg_alpha[MIN_C][T] += alpha/REPEATS;

      alpha = max_utility(cskyline, u, s, epsilon, delta, q, Qcount, Csize, MINR, stop_option, prune_option, dom_option, T);
      avg_alpha[MIN_R][T] += alpha/REPEATS;
    }
  }

  char filename[200];
  sprintf(filename, "output/vary-T-%d-%d-e%lf-d%lf-q%d-s%d-%d.dat", dim, P->numberOfPoints, epsilon, delta, q, s, time(NULL));
  FILE* fp = fopen(filename, "w");

  for(int T = MIN_T; T <= MAX_T; T += 20)
    {
      printf("%d\t%lf\t%lf\n", T, avg_alpha[MIN_C][T], avg_alpha[MIN_R][T]);
      fprintf(fp, "%d\t%lf\t%lf\n", T, avg_alpha[MIN_C][T], avg_alpha[MIN_R][T]);
    }
  fclose(fp);

}
