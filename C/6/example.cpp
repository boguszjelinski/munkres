
#include "munkres.hpp"

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

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
    const vector<std::pair<unsigned, unsigned>>& matching) noexcept
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

   ss << std::setprecision(4);
   ss << "|L|  = " << n_lhs_verts << std::endl;
   ss << "|R|  = " << n_rhs_verts << std::endl;
   ss << "cost = " << assignment_cost(cost_func, matching) << std::endl;
   for(auto r = 0u; r < n_lhs_verts; ++r) {
      for(auto c = 0u; c < n_rhs_verts; ++c) {
         if(c > 0) ss << "  ";
         const bool it_is = is_matching(r, c);
         if(it_is) ss << ANSI_COLOUR_BLUE_BG;
         ss << cost_func(r, c);
         if(it_is) ss << ANSI_COLOUR_RESET;
      }
      ss << std::endl;
   }

   return ss.str();
}

// ------------------------------------------------------------- print path info
//
template<typename T>
static void test_M(std::vector<T> M, unsigned rows, unsigned cols)
{
   auto f        = [&](unsigned r, unsigned c) { return M[r * cols + c]; };
   auto matching = munkres_algorithm<T>(rows, cols, f);
   std::cout << "Munkres (Hungarian) Algorithm:" << std::endl;
   std::cout << print_munkres_result(rows, cols, f, matching) << std::endl;
}

// ------------------------------------------------------------------------ main
//
int main(int, char**)
{
   // Unique solution
   test_M<int>({10,6,17,4,18,8,11,12,21,11,14,6,3,26,20,5,4,13,23,13,10,19,14,16,4,3,19,4,8,25,10,5,28,15,24,7,29,26,1,19,6,1,17,7,1,18,16,29,17,29,11,16,8,10,14,29,22,26,25,17,5,1,6,21,21,15,2,19,24,23,4,22,21,24,17,27,8,17,17,1,3,0,27,19,9,8,26,20,17,0,14,5,29,26,17,21,17,7,2,6}, 
   10, 10);

   // Multiple solutions
   test_M<int>({30, 25, 10, 15, 10, 20, 25, 20, 15}, 3, 3);

   // More rows than columns
   vector<double> M = {8.463,
                       4.191,
                       3.132,
                       6.852,
                       5.245,
                       2.044,
                       4.434,
                       8.781,
                       2.295,
                       0.273,
                       5.344,
                       6.704,
                       9.139,
                       4.173,
                       4.572};
   test_M<double>(M, 5, 3);
   test_M<double>(M, 3, 5); // more columns than rows

   return EXIT_SUCCESS;
}
