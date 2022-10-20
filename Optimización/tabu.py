import random

def Btabu(s, fs, x, f):
    vecindad=[]
    Ltabu=[]
    T=3
    i=1
    j=1
    iteraciones=10

    Ltabu.append(x)
    while i<iteraciones:
        print(Ltabu)
        #crear la vecindad expandida, todos los valores menos el actual
        for n in s:
            if n!=x:
                vecindad.append(n)
        for j in range(len(Ltabu)):
            if vecindad[j]==Ltabu[j]:
                vecindad.pop(j)
    
        x1=vecindad[random.randint(0, len(vecindad)-1)]
        #comparar los vecinos con el valor aleatorio inicial
        if fs[s.index(x1)]<f:
            x=x1
            Ltabu.append(x1) #agregar nueva posicion a LTabu
            f=fs[s.index(x1)]
        #eliminar indices de Ltabu apartir de valor T iteracion
        if(len(Ltabu)>T):
            Ltabu.pop(0)
        print(i)
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
    i,x,f=Btabu(S, FS, x, f)
    print("Iteraciones: ",i)
    print("x* =", x)
    print("f* =", f)
