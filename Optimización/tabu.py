import random

def Btabu(s, fs, mejor, f, iteraciones):
    vecindad=[]
    Ltabu=[]
    num_soluciones=len(s)
    T=3
    i=1
    j=1
    Ltabu.append(mejor)
    #repetir mientras
    while i<iteraciones and len(Ltabu)!=num_soluciones:
        #crear la vecindad expandida, todos los valores menos el actual
        for n in s:
            if n!=mejor:
                vecindad.append(n)
        #quitar los que se encuentran en lista tabu
        for j in range(len(Ltabu)):
            if vecindad[j]==Ltabu[j]:
                vecindad.pop(j)

        #generar un candidato
        candidato=vecindad[random.randint(0, len(vecindad)-1)]
        #evaluar un candidato
        candidato_eval=fs[s.index(candidato)]
        #comparar los vecinos con el valor aleatorio inicial
        if candidato_eval<f:
            mejor=candidato
            Ltabu.append(candidato) #agregar nueva posicion a LTabu
            f=candidato_eval
        #eliminar indices de Ltabu apartir de valor T iteracion
        if(len(Ltabu)>T):
            Ltabu.pop(0)

        vecindad=[]
        i=i+1
    return i, mejor, f

if __name__ == '__main__':
    S=[1,2,3,4,5,6,7,8,9,10] #conjunto de valores de x
    FS=[90,60,50,80,100,40,20,70,30,10] #soluciones de F(x)
    #valor aleatorio entre el primer y ultimo valor del arreglo S
    inicio=random.randint(S[0], S[-1])
    print("Inicio-> ", inicio)
    x=inicio
    #f es el indice aleatorio del arreglo en FS
    f=FS[S.index(x)]
    iter=100
    #funcion de busqueda local
    #i,x,f=BLocal(S, FS, x, f)
    i,x,f=Btabu(S, FS, x, f, iter)
    print("Iteraciones: ",i)
    print("x* =", x)
    print("f* =", f)
