//============================================================================
// Author      : Xiaokang Guo, Nov 17 2018, xg590@nyu.edu
// Copyright   : You are free to do anything about it
// Description : This is a pure C version of Dr. Robert A. Pilgrim's C# imple-
//               mentation. His original C# code is at:
//               http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
//               Minor modification and more notes are made in C code.
//               A PDF which details Robert's example is also provided at: 
//               https://github.com/xg590/munkres/blob/master/Munkres.pdf
//============================================================================

/*  This is from Robert A. Pilgrim
    The MIT License (MIT)

    Copyright (c) 2000 Robert A. Pilgrim
                       Murray State University
                       Dept. of Computer Science & Information Systems
                       Murray,Kentucky

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */

#include "stdio.h"
#include "string.h"
#include "float.h"
#include <sys/time.h>

void print_cost_matrix(double *cost, int width, int length) // {cost: pointer to cost matrix, width: row number of matrix, length: column number of matrix}
{
    printf("Cost:");
    for (int i=0; i<width; ++i)
    {
        for (int j=0; j<length; ++j)
            printf("\t%.1lf", *(cost++));
        printf("\n");
    }
}

void print_mask_matrix(int *mask, int width, int length) // {mask: pointer to mask matrix}
{
    printf("Mask:");
    for (int i=0; i<width; ++i)
    {
        for (int j=0; j<length; ++j)
            printf("\t%d", *(mask++));
        printf("\n");
    }
}

void print_vector(int *v, int n, int elapsed) // {v: pointer to vector, n: length of vector}
{
    FILE *myFile = fopen("output.txt", "w");
    fprintf(myFile, "%d\n", elapsed);
    for (int i=0; i<n; ++i)
        fprintf(myFile, "%d\n", *(v++));
    fclose(myFile);
}

void step_one(double *cost, int width, int length)
{
    double min = 0;
    for (int i=0; i<width; ++i)
    {
        min = *(cost + i*length);
        for (int j=1; j<length; ++j)
            if (*(cost + i*length + j) < min) // Find the minimum in i-th row
                min = *(cost + i*length + j);
        for (int j=0; j<length; ++j)
            *(cost + i*length + j) -= min; // Minimum subtraction
    }
}

void step_two(double *cost, int *mask, int width, int length)
{
    int row_cover[width], col_cover[length];
    memset(row_cover, 0, sizeof(int)*width);
    memset(col_cover, 0, sizeof(int)*length);
    for (int i=0; i<width; ++i)
    {
        for (int j=0; j<length; ++j)
        {
            if (*(cost+i*length+j) == 0 && *(row_cover+i) == 0 && *(col_cover+j) == 0) // Find an uncovered zero in cost matrix
            {
                *(mask+i*length+j) = 1;  // Generate the mask matrix, star zeros (put a 1 in mask matrix).
                *(row_cover+i) = 1; // Cover the row temporarily
                *(col_cover+j) = 1;
            }
        }
    }
}

int step_three(int *mask, int width, int length, int *row_cover, int *col_cover) // {row_cover: a list contains indexes of the covered rows}
{
    int count = 0;
    for (int i=0; i<width; ++i)
        for (int j=0; j<length; ++j)
            if (*(mask+i*length+j) == 1) // Cover j-th column which contains a starred zero
                *(col_cover+j) = 1;

    for (int i=0; i<length; ++i)
        if (*(col_cover+i) == 1) // Count covered column
            ++count;

    if (count >= length)
        return 7; // All columns are covered, then done.
    else
        return 4;
}

void find_a_noncovered_zero(int *row, int *col, // return row and column indexes of the non-covered zero
                 double *cost, int width, int length,
                 int *row_cover, int *col_cover)
{
    *row = -1;
    *col = -1;

    for (int i=0; i<width; ++i)
    {
        for (int j=0; j<length; ++j)
        {
            if (*(cost + i*length + j) == 0 && *(row_cover+i) == 0 && *(col_cover+j) == 0)
            {
                *row = i;
                *col = j;
                return;
            }
        }
    }
}

// methods to support step 4
int find_star_in_row(int *mask, int row, int length) // return the column index of a stared zero in the same row of the newly primed zero
{
    int col = -1;
    for (int j = 0; j < length; ++j)
        if (*(mask+row*length+j) == 1)
            col = j;
    return col;
}

int step_four(double *cost, int *mask, int width, int length, int *row_cover, int *col_cover, int *path_row_0, int *path_col_0) // {path_row_0: row index of the starting point of the zig-zag path in step 5}
{
    int row = -1;
    int col = -1;
    int col_buff = -1;
    while (1)
    {
        find_a_noncovered_zero(&row, &col, cost, width, length, row_cover, col_cover);
//        printf("non-covered zero: (%d, %d)\n", row, col);
        if (row == -1)
        { // no non-covered zeros left
            return 6;
        }
        else
        { // found a non-covered zero,
            *(mask+row*length+col) = 2;  // Prime the non-covered zero in the mask matrix (put a 2 in mask matrix)
            col_buff = find_star_in_row(mask, row, length); // Find a starred zero in the same row
            if (col_buff > -1) // found an starred zero
            { // Cover this row and uncover the column containing the starred zero
                *(row_cover+row) = 1;
                *(col_cover+col_buff) = 0;
            }
            else
            { // Once there is no starred zero in the row containing this primed zero, Go to Step 5.
                *path_row_0 = row; // A starting point for a zig-zag path covering primed and starred zeros
                *path_col_0 = col;
                return 5;
            }
        }
    }
}

// methods to support step 5
int find_star_in_col(int *mask, int col, int length) // return row index
{
    for (int i = 0; i < length; ++i)
        if (*(mask+i*length+col) == 1)
            return i;
    return -1;
}

int find_prime_in_row(int *mask, int row, int length)
{
    for (int j = 0; j < length; ++j)
        if (*(mask+row*length+j) == 2)
            return j;
    return -1;
}

void augment_path(int *mask, int length, int *path, int count) // primed to starred and starred to unmarked
{                                        // { path: a list shows the zig-zag path contains primed and starred zeros, path: how many zeros in the path}
    int path_i_0 = 0;
    int path_i_1 = 0;

    for (int i = 0; i < count; ++i)
    {
        path_i_0 = *(path + i*2);
        path_i_1 = *(path + i*2 + 1);
        if (*(mask + path_i_0*length + path_i_1) == 1)
            *(mask + path_i_0*length + path_i_1) = 0;
        else
            *(mask + path_i_0*length + path_i_1) = 1;
    }
}

void clear_covers(int width, int length, int *row_cover, int *col_cover)
{
    for (int i = 0; i < width; ++i)
    {
        *(row_cover+i) = 0;
    }
        for (int i = 0; i < length; ++i)
        {
        *(col_cover+i) = 0;
    }
}

void erase_primes(int *mask, int width, int length)
{
    for (int i=0; i<width; ++i)
        for (int j=0; j<length; ++j)
            if (*(mask+i*length+j) == 2)
                *(mask+i*length+j) = 0;
}

int step_five(int *mask, int width, int length, int *row_cover, int *col_cover, int *path_row_0, int *path_col_0)
{
    int row = -1;
    int col = -1;
    int count = 1;
    int path[width*length];
    memset(path, 0, sizeof(int)*width*length);
    *(path + count*2 - 2) = *path_row_0; // The result from step 4 becomes the first element in a list called path (starting point in path)
    *(path + count*2 - 1) = *path_col_0;
//    printf("starting zero for the zig-zag path: (%d, %d)\n", *path_row_0, *path_col_0);
    while (1)
    {
        row = find_star_in_col(mask, *(path + count*2 - 1), length); // Is a starred zero within the column having the starting zero or primed zero (found in following 11th lines)
        if (row > -1)
        { // yes and put the new starred zero in the list
            ++count;
            *(path + count*2 - 2) = row;
            *(path + count*2 - 1) = *(path + count*2 - 3);
        }
        else
        { // no
            break;
        }
        col = find_prime_in_row(mask, *(path + count*2 - 2), length); // Find the primed zero within the row of starred zero(, which will be tested for a new starred zero above)
        ++count;
        *(path + count*2 - 2) = *(path + count*2 - 4);
        *(path + count*2 - 1) = col;
    }
    augment_path(mask, length, path, count);
    clear_covers(width, length, row_cover, col_cover);
    erase_primes(mask, width, length);
    return 3;
}

//methods to support step 6
double find_smallest(double *cost, int width, int length, int *row_cover, int *col_cover) // smallest non-covered number
{
    double min = DBL_MAX;
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < length; ++j)
            if (*(row_cover+i) == 0 && *(col_cover+j) == 0)
                if (min > *(cost + i*length + j))
                    min = *(cost + i*length + j);
    return min;
}

int step_six(double *cost, int width, int length, int *row_cover, int *col_cover)
{
    double min = find_smallest(cost, width, length, row_cover, col_cover); // smallest non-covered number
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            if (*(row_cover+i) == 1) // covered row unaffected
                *(cost + i*length + j) += min;
            if (*(col_cover+j) == 0)
                *(cost + i*length + j) -= min; // affect the non-covered column
        }
    }
//    printf("zero maker works for the noncovered\n");
    return 4;
}

double total_cost(double *cost, int *mask, int width, int length)
{
    double sum = 0;
    for (int i=0; i<width; ++i)
        for (int j=0; j<length; ++j)
            if (*(mask+i*length+j) == 1)
                sum += *(cost+i*length+j);
    return sum;
}

void get_assignment(int *mask, int width, int length, int *matched_col, int *matched_row)
{
    memset(matched_col, -1, sizeof(int)*width);
    memset(matched_row, -1, sizeof(int)*length);
    for (int i=0; i<width; ++i)
        for (int j=0; j<length; ++j)
            if (*(mask+i*length+j) == 1)
            {
                *(matched_col+i) = j;
                *(matched_row+j) = i;
            }
    return;
}

int munkres(double *cost_matrix, int width, int length, int *matched_col, int *matched_row)
{//  matched_col: a list shows the assigned column index for every row.
 //  For example, {3, 1, 0, 2} means the owner of the 3rd column is matched to the owner of the 0th row,
 //                                                   1st column            to                  1st row,
 //                                                   0th column            to                  2nd row,
 //                                               and 2nd column            to                  3rd row.
//    double cost_matrix_copy[width * length];
//    memcpy(cost_matrix_copy, cost_matrix, sizeof(double)*width*length);
    int mask_matrix[width * length];
    memset(mask_matrix, 0, sizeof(int)*width*length);
    int row_cover[width];
    memset(row_cover, 0, sizeof(int)*width);
    int col_cover[length];
    memset(col_cover, 0, sizeof(int)*length);

//    printf("Step 1 begins...\n");
    step_one(cost_matrix, width, length);
//    printf("Step 1 ends, step 2 next...\nStep 2 begins...\n");
    step_two(cost_matrix, mask_matrix, width, length);
//    printf("Step 2 ends, step 3 next...\n");
    int step = 3;
    int path_row_0;
    int path_col_0;
    while (1)
    {
//        print_cost_matrix(cost_matrix, width, length);printf("\n");
//        print_mask_matrix(mask_matrix, width, length);printf("\n");
//        printf("row_cover");print_vector(row_cover, width);
//        printf("col_cover");print_vector(col_cover, length);
        switch (step)
        {
            case 3:
//                printf("-----------------------\nStep 3 begins...\n");
                step = step_three(mask_matrix, width, length, row_cover, col_cover);
//                printf("Step 3 ends, step %d next...\n\n", step);
                break;
            case 4:
//                printf("-----------------------\nStep 4 begins...\n");
                step = step_four(cost_matrix, mask_matrix, width, length, row_cover, col_cover, &path_row_0, &path_col_0);
//                printf("Step 4 ends, step %d next...\n\n", step);
                break;
            case 5:
//                printf("-----------------------\nStep 5 begins...\n");
                step = step_five(mask_matrix, width, length, row_cover, col_cover, &path_row_0, &path_col_0);
//                printf("Step 5 ends, step %d next...\n\n", step);
                break;
            case 6:
//                printf("-----------------------\nStep 6 begins...\n");
                step = step_six(cost_matrix, width, length, row_cover, col_cover);
//                printf("Step 6 ends, step %d next...\n\n", step);
                break;
            case 7:
                get_assignment(mask_matrix, width, length, matched_col, matched_row);
//              ;double result = total_cost(cost_matrix_copy, mask_matrix, width, length);
//                printf("----------------------------\nMinimum total cost is %.2lf.\n", result);
                return 0;
        }
    }
    return 1;
}

int main()
{
    int width  = 5;
    int length = 4;
    /*double cost_matrix[] =
                         {10, 19, 8, 15,
                          10, 18, 7, 17,
                          13, 16, 9, 14,
                          12, 19, 8, 18,
                          14, 17,10, 19};
    */
    FILE *myFile;
    myFile = fopen("/home/m91127/munkres/input.txt", "r");
    if (myFile == NULL){
        printf("Error Reading File\n");
        return 0;
    }
    fscanf(myFile, "%d", &width);
    fscanf(myFile, "%d", &length);
    double cost_matrix[length*width];
    int value;
    for (int i = 0; i < width*length; i++) {
        fscanf(myFile, "%d", &value);
        cost_matrix[i] = value;
    }
    fclose(myFile);

    int matched_col[width];
    int matched_row[length];
    struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);

    munkres(cost_matrix, width, length, matched_col, matched_row);

    gettimeofday (&tvalAfter, NULL);
    int millis = (((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L +tvalAfter.tv_usec) - tvalBefore.tv_usec)/1000; 
    print_vector(matched_col, width, millis);
    //printf("matched_row:");print_vector(matched_row, length);
    return 0;
}
