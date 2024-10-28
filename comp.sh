#!/bin/bash
cd C/1
cc munkres.c -O3 -o ../../munkres1
cd ../2
g++ -w testMain.cpp Hungarian.cpp -O3 -o ../../munkres2
cd ../3
g++ -std=c++11 -O3 hungarian.cpp -o ../../munkres3
cd ../4
gcc -w -O3 munk.c -o ../../munkres4
cd ../5
g++ -w lap.cpp -O3 -o ../../lap1
cd ../6
g++ munk.cpp -O3 -o ../../munkres6

