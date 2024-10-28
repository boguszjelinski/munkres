import datetime
import numpy as np
from cvxopt.glpk import ilp
from cvxopt import matrix

def solve (n, cost): 
    if n==0: return 0, []
    c = matrix(cost, tc='d')
    # constraints
    arr = np.zeros((n*2, n*n)) 
    for i in range(0,n):
        for j in range(0,n): 
            arr[i][n*i+j]=1.0
            arr[n+i][n*j+i]=1.0
    a=matrix(arr, tc='d') 
    g=matrix([ [0 for x in range(n*n)] ], tc='d')
    b=matrix(1*np.ones(n*2))
    h=matrix(0*np.ones(1))
    I=set(range(n*n))
    B=set(range(n*n))
    (status,x)=ilp(c,g.T,h,a,b,I,B)
    return x
################################################################

with open("input.txt") as f:
    nn = int(f.readline())
    cost = [[int(x) for x in line.split()] for line in f]
a = datetime.datetime.now()

x = solve(nn, cost)

b = datetime.datetime.now()
c = b - a
millis = int(c.total_seconds() * 1000)
f = open("output.txt", "w")
f.write ("%d\n" % (millis))
for i in range(0,nn*nn): 
    f.write ("%d\n" % (x[i]))
f.close()
