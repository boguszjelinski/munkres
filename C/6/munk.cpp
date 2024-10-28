
#include "munkres.hpp"

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <sys/time.h>
using std::string;
using std::vector;
using namespace std;

// ------------------------------------------------------------- assignment cost
//
template<typename T>
static T
assignment_cost(std::function<T(unsigned l, unsigned r)> cost_func,
                const vector<std::pair<unsigned, unsigned>>& matching) noexcept
{
   T cost = T(0);
   for(const auto& m : matching) cost += cost_func(m.first, m.second);
   return cost;
}

// -------------------------------------------------------- print munkres result
//
static string print_munkres_result(
    const unsigned n_lhs_verts,
    const unsigned n_rhs_verts,
    std::function<double(unsigned l, unsigned r)> cost_func,
    const vector<std::pair<unsigned, unsigned>>& matching,
    int elapsed) noexcept
{
   static const char* ANSI_COLOUR_BLUE_BG = "\x1b[44m";
   static const char* ANSI_COLOUR_RESET   = "\x1b[0m";

   std::stringstream ss("");

   auto is_matching = [&](const unsigned r, const unsigned c) {
      const auto ii
          = std::find_if(begin(matching), end(matching), [&](const auto& x) {
               return (x.first == r and x.second == c);
            });
      return ii != end(matching);
   };

   ofstream outfile;
   outfile.open("output.txt");
	outfile << elapsed << "\n";
   //ss << std::setprecision(4);
   //ss << "cost = " << assignment_cost(cost_func, matching) << std::endl;
   for(auto r = 0u; r < n_lhs_verts; ++r) {
      for(auto c = 0u; c < n_rhs_verts; ++c) {
         const bool it_is = is_matching(r, c);
         if(it_is) outfile << c << std::endl;
      }
   }

   outfile.close();
   return ss.str();
}

// ------------------------------------------------------------- print path info
//
template<typename T>
static void test_M(std::vector<T> M, unsigned rows, unsigned cols)
{
   auto f        = [&](unsigned r, unsigned c) { return M[r * cols + c]; };
   struct timeval tvalBefore, tvalAfter;
   gettimeofday (&tvalBefore, NULL);

   auto matching = munkres_algorithm<T>(rows, cols, f);

   gettimeofday (&tvalAfter, NULL);
   int millis = (((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L +tvalAfter.tv_usec) - tvalBefore.tv_usec)/1000; 

   print_munkres_result(rows, cols, f, matching, millis);
}

// ------------------------------------------------------------------------ main
//
int main(int, char**)
{
   int no_rows;
   int no_cols;

   ifstream is("input.txt");
   is >> no_rows;
   is >> no_cols;
   std::istream_iterator<int> start(is), end;
   vector<int> cost(start, end);
   is.close();

   test_M<int>(cost, no_rows, no_cols);

   return EXIT_SUCCESS;
}
