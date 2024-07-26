h = int(input('Enter height of the diamond:'))
for i in range(0,h):
    print(" " * (h-i-1) + '*' * (2*i + 1))
# print(h)

for i in range(h-1, 0, -1):
    print(' ' * (h-i) + '*' * ((i-1)*2+1))
