import random


for i in range(10):
    a= random.randint(1,10)
    b= random.randint(1,10)
    print('Question'+ str(i+1)+':',a,'*',b,'=')
    c=input()
    if c==a*b and type(c)==int:
        print('Correct')
    else:
        print('Wrong. The answer is', a*b)
