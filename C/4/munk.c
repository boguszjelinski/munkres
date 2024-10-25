//
//(Minimum) Assignment Problem by Hungarian Algorithm
//taken from Knuth's Stanford Graphbase
//
#include <stdio.h>
#include<stdlib.h>

#define INF (0x7FFFFFFF)

int size1=0;
int size2=0;

#define SIZE 2001
#define verbose (1)

int Array[SIZE][SIZE];
char Result[SIZE][SIZE];  // used as boolean

void initArray()
{
  int i,j;

  for (i=0;i<SIZE;++i)
    for (j=0;j<SIZE;++j)
      Array[i][j]=99999999;
}

void hungarian()
{
int i,j;
int false=0,true=1;

unsigned int m=size1,n=size2;
int k;
int l;
int s;
int col_mate[size1]; memset(col_mate, 0, sizeof(col_mate));
int row_mate[size2]; memset(row_mate, 0, sizeof(row_mate));
int parent_row[size2]; memset(parent_row, 0, sizeof(parent_row));
int unchosen_row[size1]; memset(unchosen_row, 0, sizeof(unchosen_row));
int t;
int q;
int row_dec[size1]; memset(row_dec, 0, sizeof(row_dec));
int col_inc[size2]; memset(col_inc, 0, sizeof(col_inc));
int slack[size2]; memset(slack, 0, sizeof(slack));
int slack_row[size2]; memset(slack_row, 0, sizeof(slack_row));
int unmatched;
int cost=0;

for (i=0;i<size1;++i)
  for (j=0;j<size2;++j)
    Result[i][j]=false;

// Begin subtract column minima in order to start with lots of zeroes 12

for (l=0;l<n;l++)
{
  s=Array[0][l];
  for (k=1;k<n;k++)
    if (Array[k][l]<s)
      s=Array[k][l];
  cost+=s;
  if (s!=0)
    for (k=0;k<n;k++)
      Array[k][l]-=s;
}
// End subtract column minima in order to start with lots of zeroes 12

// Begin initial state 16
t=0;
for (l=0;l<n;l++)
{
  row_mate[l]= -1;
  parent_row[l]= -1;
  col_inc[l]=0;
  slack[l]=INF;
}
for (k=0;k<m;k++)
{
  s=Array[k][0];
  for (l=1;l<n;l++)
    if (Array[k][l]<s)
      s=Array[k][l];
  row_dec[k]=s;
  for (l=0;l<n;l++)
    if (s==Array[k][l] && row_mate[l]<0)
    {
      col_mate[k]=l;
      row_mate[l]=k;
      goto row_done;
    }
  col_mate[k]= -1;
  unchosen_row[t++]=k;
row_done:
  ;
}
// End initial state 16
 
// Begin Hungarian algorithm 18
if (t==0)
  goto done;
unmatched=t;
while (1)
{
  q=0;
  while (1)
  {
    while (q<t)
    {
      // Begin explore node q of the forest 19
      {
        k=unchosen_row[q];
        s=row_dec[k];
        for (l=0;l<n;l++)
          if (slack[l])
          {
            int del;
            del=Array[k][l]-s+col_inc[l];
            if (del<slack[l])
            {
              if (del==0)
              {
                if (row_mate[l]<0)
                  goto breakthru;
                slack[l]=0;
                parent_row[l]=k;
                unchosen_row[t++]=row_mate[l];
              }
              else
              {
                slack[l]=del;
                slack_row[l]=k;
              }
          }
        }
      }
      // End explore node q of the forest 19
      q++;
    }
 
    // Begin introduce a new zero into the matrix 21
    s=INF;
    for (l=0;l<n;l++)
      if (slack[l] && slack[l]<s)
        s=slack[l];
    for (q=0;q<t;q++)
      row_dec[unchosen_row[q]]+=s;
    for (l=0;l<n;l++)
      if (slack[l])
      {
        slack[l]-=s;
        if (slack[l]==0)
        {
          // Begin look at a new zero 22
          k=slack_row[l];
          if (verbose)
            printf(
              "Decreasing uncovered elements by %d produces zero at [%d,%d]\n",
              s,k,l);
          if (row_mate[l]<0)
          {
            for (j=l+1;j<n;j++)
              if (slack[j]==0)
                col_inc[j]+=s;
            goto breakthru;
          }
          else
          {
            parent_row[l]=k;
            if (verbose)
              printf("node %d: row %d==col %d--row %d\n",t,row_mate[l],l,k);
            unchosen_row[t++]=row_mate[l];
          }
          // End look at a new zero 22
        }
      }
      else
        col_inc[l]+=s;
    // End introduce a new zero into the matrix 21
  }
breakthru:
  // Begin update the matching 20
  while (1)
  {
    j=col_mate[k];
    col_mate[k]=l;
    row_mate[l]=k;
    if (j<0)
      break;
    k=parent_row[j];
    l=j;
  }
  // End update the matching 20
  if (--unmatched==0)
    goto done;
  // Begin get ready for another stage 17
  t=0;
  for (l=0;l<n;l++)
  {
    parent_row[l]= -1;
    slack[l]=INF;
  }
  for (k=0;k<m;k++)
    if (col_mate[k]<0)
    {
      if (verbose)
        printf("node %d: unmatched row %d\n",t,k);
      unchosen_row[t++]=k;
    }
  // End get ready for another stage 17
}
done:

// Begin doublecheck the solution 23
for (k=0;k<m;k++)
  for (l=0;l<n;l++)
    if (Array[k][l]<row_dec[k]-col_inc[l])
      exit(0);
for (k=0;k<m;k++)
{
  l=col_mate[k];
  if (l<0 || Array[k][l]!=row_dec[k]-col_inc[l])
    exit(0);
}
k=0;
for (l=0;l<n;l++)
  if (col_inc[l])
    k++;
if (k>m)
  exit(0);
// End doublecheck the solution 23
// End Hungarian algorithm 18

for (i=0;i<m;++i)
{
  Result[i][col_mate[i]]=true;
 /*TRACE("%d - %d\n", i, col_mate[i]);*/
}
for (k=0;k<m;++k)
{
  for (l=0;l<n;++l)
  {
    /*TRACE("%d ",Array[k][l]-row_dec[k]+col_inc[l]);*/
    Array[k][l]=Array[k][l]-row_dec[k]+col_inc[l];
  }
  /*TRACE("\n");*/
}
for (i=0;i<m;i++)
  cost+=row_dec[i];
for (i=0;i<n;i++)
  cost-=col_inc[i];
//printf("Cost is %d\n",cost);
}

main()
{
int y,x,i;
FILE *myFile;

initArray();

myFile = fopen("input.txt", "r");
if (myFile == NULL){
    printf("Error Reading File\n");
    return 0;
}
fscanf(myFile, "%d", &size1);
fscanf(myFile, "%d", &size2);
double cost_matrix[size2*size1];
int value;
for (int i = 0; i < size1; i++)
  for (int j = 0; j< size1; j++) {
    fscanf(myFile, "%d", &value);
    Array[i][j] = value;
}
fclose(myFile);

hungarian();

FILE *outFile = fopen("output.txt", "w");
for (y=0;y<size1;++y)
  for (x=0;x<size2;++x)
    if (Result[y][x])
      fprintf(outFile, "%d\n", x);
fclose(outFile);
}