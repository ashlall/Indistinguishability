all:
	g++ -w *.c *.cpp -lglpk -lm -o run

valgrind:
	g++ -w *.c *.cpp -g -O0 -lglpk -lm -o run

clean:
	rm run