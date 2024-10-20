from munkres import Munkres

matrix = [[10, 19, 8, 15],
        [10, 18, 7, 17],
        [13, 16, 9, 14],
        [12, 19, 8, 18],
        [14, 17,10, 19]]
m = Munkres()
indexes = m.compute(matrix)
for row, column in indexes:
    print(f'{row} {column}')
