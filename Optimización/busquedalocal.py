import random

def BLocal(s, fs, x, f):
    vecindad=[]
    i=0
    continuar=True #variable de control 
    
    while continuar:
        #crear la vecindad en el rango x-1, x+1+1
        for n in range(x-1, x+1+1): 
            if n in S and n!=x:
                vecindad.append(n)

        continuar=False
        #comparar los vecinos con el valor aleatorio inicial
        for n in vecindad:
            if FS[S.index(n)]<f:
                x=n
                f=FS[S.index(n)]
                continuar=True
        vecindad=[]
        i=i+1
    return i, x, f
    
def BLocalExpAle(s, fs, x, f):
    vecindad=[]
    i=1
    iteraciones=10
    
    while i<iteraciones:
        #crear la vecindad expandida, todos los valores menos el actual 
        for n in s: 
            if n!=x:
                vecindad.append(n)
                
        x1=vecindad[random.randint(0, len(vecindad)-1)]
        #comparar los vecinos con el valor aleatorio inicial
        if fs[s.index(x1)]<f:
            x=x1
            f=fs[s.index(x1)]
        print(x, f)
        vecindad=[]
        i=i+1
    return i, x, f

if __name__ == '__main__':
    S=[1,2,3,4,5,6,7,8] #conjunto de valores de x
    FS=[90,60,50,80,100,40,20,70] #soluciones de F(x)
    
    #valor aleatorio entre el primer y ultimo valor del arreglo S
    inicio=random.randint(S[0], S[-1]) 
    print("Inicio-> ", inicio)
    x=inicio
    #f es el indice aleatorio del arreglo en FS
    f=FS[S.index(x)] 
    #funcion de busqueda local 
    #i,x,f=BLocal(S, FS, x, f)
    i,x,f=BLocalExpAle(S, FS, x, f)
    print("Iteraciones: ",i)
    print("x* =", x)
    print("f* =", f)
