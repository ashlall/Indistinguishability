# Indistinguishability
Readme for code for 'The Indistinguishability Query'

Base code by Min Xie (first author of Strongly Truthful Interactive Regret Minimization, SIGMOD 2019).

This repository has the following new algorithms:
1. Squeeze-u
2. Squeeze-u2
3. MinR
4. MinD

This package also provides the datasets used.

INSTALLATION:

1. Make sure there is a folder called "input/" and a folder "output/" under the working directory. They will be used for storing the input/output files and some intermediate results.
2.  You will need to install the GLPK package (for solving LPs) at first.
        See GLPK webpage <http://www.gnu.org/software/glpk/glpk.html>.
    The Makefile should handle all compilation (once you get GLPK installed correctly).
        Note that "-lglpk" in the Makefile is the GLPK package and you may need to change it to the location where you installed the GLPK package.
3. Run 'make' to compile code.

REPRODUCIBILITY:

To reproduce the experiments in the paper 'The Indistinguishability Query' please consult EXPERIMENTS.txt for exact command line options. For example,

./run Q a3-100000.txt 0.05 0.0 3 9 B 0

will run the code for varying the number of questions asked of the user (q) on the data set a3-100000.txt (in the input/ folder) with epsilon=0.5, delta=0.0, s=3 tuples per round, running (B)oth real and artificial tuple queries, and a random seed of 0. While any random seed may be used, the experiments consistently used a seed value of 0 for reproducibility.

