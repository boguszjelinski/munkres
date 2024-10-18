#include <iostream>
#include "Hungarian.h"
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

void read_data(ifstream& , vector <vector <double> >&, int, int );
void print_data(const vector <vector<double> >&, int, int);

void read_data(ifstream& is, vector <vector <double> >& m, int rows, int cols) {
    double buf = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            is >> buf;
            m[row][col] = buf;
        }
    }
}

void print_data(const vector <vector<double> >& m, int rows, int cols){
    cout << rows << "\n";
    cout << cols << "\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << std::setw(6) << m[i][j];
        }
        cout << endl;
    }
}

int main(void)
{
    // please use "-std=c++11" for this initialization of vector.
	/*vector< vector<double> > costMatrix =  {
				           	{10, 19, 8, 15},
            	            {10, 18, 7, 17},
                	        {13, 16, 9, 14},
                    	    {12, 19, 8, 18},
                        	{14, 17,10, 19}};
 */
  	int no_rows;
    int no_cols;
        
    ifstream infile;
    infile.open("input.txt");
    infile >> no_rows;
	infile >> no_cols;
    vector<vector<double>> costMatrix(no_rows, vector<double>(no_cols));

    read_data(infile, costMatrix, no_rows, no_cols);
    infile.close();
    //print_data(costMatrix, no_rows, no_cols);

	HungarianAlgorithm HungAlgo;
	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

    ofstream outfile;
    outfile.open("output.txt");
	for (unsigned int x = 0; x < costMatrix.size(); x++)
		outfile << assignment[x] << "\n";
	//std::cout << cost << std::endl;
    outfile.close();
	return 0;
}
