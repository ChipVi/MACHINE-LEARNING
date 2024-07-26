n = int(input('Enter index letter:'))
num= 96 + n

for i in range(0,n):
    print("-" * (2*(n-1)-2*i), end='')
    for sub in range(0,i):
        print(chr(num - sub), end='-')
    for sub in range(i,-1,-1):
        if (sub == 0):
            print(chr(num - sub), end='')
        else: print(chr(num - sub), end='-')
    print("-" * (2*(n-1)-2*i))

for i in range(n-2, -1, -1):
    print("-" * (2*(n-1)-2*i), end='')
    for sub in range(0,i):
        print(chr(num - sub), end='-')
    for sub in range(i,-1,-1):
        if (sub == 0):
            print(chr(num - sub), end='')
        else: print(chr(num - sub), end='-')
    print("-" * (2*(n-1)-2*i))