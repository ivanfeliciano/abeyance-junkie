---
layout: post
title:  "Redes neuronales artificiales"
date:   mar dic 18 17:10:04 CST 2018
---


En su forma más general, una *red neuronal artificial* o simplemente
*red neuronal*, es una máquina diseñada para modelar la manera en la
cual el cerebro realiza una tarea en particular o una función de
interés. Las redes neuronales artificiales utilizan una enorme cantidad
de interconexiones de células de cómputo conocidas como neuronas o
unidades de procesamiento.

Una red neuronal artificial es un procesador paralelo distribuido
construido de unidades simples de procesamiento, que tiene una
naturaleza propensa a almacenar conocimiento basado en experiencias y
poniéndolo a disposición para su uso. Se asemeja al cerebro en dos
aspectos:

1.  El conocimiento es adquirido por la red desde su ambiente a través
    de un proceso de aprendizaje. El proceso de aprendizaje se hace a
    través de un algoritmo de aprendizaje.

2.  Las fuerza de conexión entre neuronas, también conocidas como pesos
    sinápticos, son usadas para almacenar el conocimiento adquirido
    [@johnahertzandersskroghrichardgpalmer1991].

En términos generales, una red neuronal artificial consiste de un gran
número de procesadores simples enlazados por conexiones ponderadas. Cada
unidad recibe entradas que vienen de muchas otras unidades y produce un
valor escalar como salida que depende sólo de la información disponible
localmente, ya sea de la que guarda internamente o la que llega a través
de las conexiones ponderadas. La información es distribuida y actúa como
entrada a otras unidades de procesamiento.

Por sí mismo, un elemento de procesamiento no es muy poderoso; el poder
del sistema surge de la combinación de muchas unidades en una red. Una
red está especializada en implementar diferentes funciones cambiando la
topología de las conexiones en la red y los valores de las conexiones
ponderadas.

Las unidades de procesamiento tienen respuestas como

$$\label{eq:neuron-output}
y = f\bigg (\sum_k w_k x_k\bigg )$$

donde $x_k$ son las señales de salida de otros nodos o entradas de un
sistema externo, $w_k$ son los pesos de los enlaces de conexión y $f$ es
una función no lineal. Aquí una la unidad calcula una combinación lineal
de los pesos y sus entradas y la pasa por $f$ para producir un valor
escalar. $f$ es conocida como *función de activación*, y es muy común
que se utilice una función no lineal acotada y creciente como la
sigmoide, definida como sigue

$$f(u) = \frac{1}{1 + e^{-u}}$$

\centering
 \[scale=0.30\]/assets/neuron.png

El término perceptrón a menudo se utiliza para referirse a cualquier red
de nodos *feedforward* con respuestas como la ecuación
([\[eq:neuron-output\]](#eq:neuron-output){reference-type="ref"
reference="eq:neuron-output"}). Una red puede tener cualquier estructura
arbitraria, sin embargo, las arquitecturas en capas son muy populares.\

El perceptrón multicapa o MLP (por sus siglas en inglés), es ampliamente
utilizado. En tales estructuras las unidades en la capa $L$ reciben
entradas de la capa que les precede $L-1$ y envía sus salidas a la capa
siguiente $L+1$. Entradas externas se presentan en la primera capa y las
salidas del sistema se toman de la última capa. Las capas internas se
llaman *capas ocultas*. Las redes más simples tienen una sola capa
activa, las de las unidades de salida, por convención, las entradas no
se cuentan como una capa activa ya que no realizan algún tipo de
procesamiento. Las redes con una sola capa son menos poderosas que las
multicapas por lo que sus aplicaciones están muy limitadas.

\centering
![Una red neuronal feedforward completamente conectada con múltiples
capas.[]{label="fig:mpl"}](/assets/mlp.png)

Que una estructura sea *feedforward* o *prealimentada* significa que no
existen bucles en la red, la información siempre alimenta hacia
adelante, nunca hacia atrás. La red implementa un mapeo estático que
depende únicamente de las entradas que se presenten y es independiente
de los estados previos del sistema.\

En una red *completamente conectada*, cada nodo en la capa $L$ recibe
entradas de cada nodo en la capa anterior $L-1$ y envía su salida a
todos los nodos en la capa $L+1$.

Las redes neuronales pueden verse como gráficas cuyos nodos son unidades
de cálculo y cuyas aristas transmiten información numérica de nodo a
nodo. Cada unidad de cómputo es capaz de evaluar su entrada en una
función de activación. La red representa una cadena de funciones
compuestas, que transforman una entrada en un vector de salida. La red
es una implementación particular de una función compuesta desde la
entrada hasta el espacio de salida, a la cual llamamos *función de red*.

En el contexto del aprendizaje automático supervisado, la experiencia de
la que aprenden las redes neuronales es un conjunto con patrones de
entrada donde cada patrón tiene una salida deseada. Nuestro problema de
aprendizaje consiste en encontrar la combinación de pesos óptima tal que
la función de red $\varphi$ se aproxime lo más que se pueda a una
función $F$. Sin embargo, no contamos con la función $F$ explícitamente,
sino sólo a través de algunas muestras.

Consideremos una red prealimentada con $n$ unidades de entrada y $m$
unidades de salida. Ésta consiste de cualquier número de unidades
ocultas y de cualquier topología en sus conexiones. También contamos con
un conjunto de entrenamiento
$\{(\mathbf{x}_1, \mathbf{t}_1), \dots, (\mathbf{x}_p, \mathbf{t}_p)\}$,
con $p$ pares ordenados de vectores de tamaño $n$ y $m$, los cuales se
llaman patrones de entrada y de salida. Definamos a las funciones de
activación de cada nodo en la red como continuas y diferenciables. Los
pesos de las aristas son números reales seleccionados aleatoriamente.
Cuando el patrón de entrada $\mathbf{x}_i$ del conjunto de entrenamiento
se presenta en la red, produce una salida $\mathbf{y}_i$, generalmente
diferente al objetivo $\mathbf{t}_i$. Queremos que $\mathbf{y}_i$ y
$\mathbf{t}_i$ sean idénticos para $i = 1, \dots, p$, usando un
algoritmo de aprendizaje. De manera más precisa, queremos minimizar una
*función de costo* de la red con respecto a sus pesos; por ejemplo la
definida como sigue

$$E = \frac{1}{2} \sum_{i=1}^p \|\mathbf{t}_i - \mathbf{y}_i\|^2$$

Después de minimizar esta función para el conjunto de entrenamiento,
nuevos patrones de entrada desconocidos alimentan a la red y esperamos
que se interpolen. La red debe reconocer cuando un nuevo vector de
entrada es similar a los patrones aprendidos y producir una salida
semejante.

El *algoritmo de propagación hacia atrás o retropropagación* es el
método para entrenar redes neuronales más utilizado y se describe en la
siguiente sección.

### Retropropagación

El término retropropagación se refiere a dos cosas diferentes. Primero,
describe un método para calcular las derivadas de la función de costo
con respecto a los pesos aplicando la regla de la cadena. Segundo,
describe un algoritmo de entrenamiento, equivalente al descenso del
gradiente; el gradiente de la función de costo es calculado y usado para
corregir los pesos iniciales, que fueron elegidos aleatoriamente.

Como algoritmo de entrenamiento, el propósito de la retropropagación es
ajustar los pesos de la red para producir la salida deseada como
resultado a cada patrón de entrada un un conjunto de patrones de
entrenamiento. Es un algoritmo *supervisado* en el sentido que, para
cada patrón de entrada, existe exactamente una salida correcta.

Para entrenar una red, es necesario tener un conjunto de patrones de
entrada y sus salidas deseadas correspondientes, además una función de
error que mide el "costo" de las diferencias entre las salidas de la red
y los valores deseados. De manera muy general, los pasos básicos del
algoritmo son los siguientes:

1.  Inicializar los pesos de manera aleatoria con valores pequeños.

2.  Alimentar a la red con una muestra del conjunto de entrenamiento
    para obtener las salidas. Este paso también es conocido como
    feedforward o propagación hacia adelante.

3.  Comparar las salidas con los valores deseados y calcular el error.

4.  Calcular las derivadas del error con respecto a cada uno de los
    pesos $\frac{\partial E}{\partial w_{ij}}$.

5.  Ajustar los pesos para minimizar el error.

6.  Repetir.

#### Propagación hacia adelante

Por simplicidad, supongamos que los nodos están indexados tal que
$i < j$ implica que el nodo $j$ sigue al nodo $i$ en términos de
dependencia. Esto quiere decir que, el estado del nodo $j$ puede
depender, quizás indirectamente, del estado del nodo $i$, pero el nodo
$i$ no depende del nodo $j$. Esta notación permite que durante las
simulaciones se evite la necesidad de lidiar con cada capa de manera
separada, al hacer un seguimiento de los índices de la capa. Por
supuesto, este esquema de indexado es compatible con las estructuras
multicapas.

En el paso hacia adelante, la red calcula una salida basada en sus
entradas actuales. Cada nodo $j$ calcula una suma ponderada $a_j$ de sus
entradas y la pasa a través de una función para obtener la salida del
nodo $y_j$.

$$\label{linear_combination_aj}
a_j = \sum_{i < j} w_{ij} y_i$$

$$y_j = f(a_j)$$

$w_{ij}$ denota el peso que llega al nodo $j$ desde el nodo $i$. El
índice $i$ en la suma va sobre todos los índices $i < j$ de nodos que
envían una entrada al nodo $j$. Normalmente la función $f$ es la
sigmoide, sin embargo, no es la única función de activación.

Cada nodo es evaluado en orden, comenzando con el primer nodo oculto y
continuando hasta llegar al último nodo de salida. En redes con
múltiples capas, la primera capa oculta se actualiza basándose en las
salidas de los nodos de entrada, que son los valores de un vector de
características de una muestra; la segunda capa oculta se actualiza
basándose en las salidas de la primera capa oculta, y se continúa así
hasta llegar a la capa de salida la cual se actualiza con las salidas de
la última capa oculta.

#### Cálculo del error

A menos que la red esté perfectamente entrenada, las salidas de la red
diferirán de las salidas deseadas. Como ya vimos, para medir esa
diferencia, se utiliza una función de costo, que por ahora será la *suma
de cuadrados del error* o *SSE* (por sus siglas en inglés).

$$E = \frac{1}{2} \sum_p\sum_k(t_{pk} - y_{pk})^2$$

donde $p$ indexa a todos los patrones del conjunto de entrenamiento, $k$
indexa a los nodos de salida, $t_{pk}$ y $y_{pk}$ son respectivamente,
el objetivo y la salida actual de la red para el $k$ ésimo nodo de
salida de la muestra $p$. Una de las razones por las que la SSE es
conveniente es porque los errores entre las diferentes muestras o
patrones y las diferentes salidas son independientes, el error total es
la suma de los errores cuadrados individuales.

$$E = \sum_p E_p$$ donde
$$E_p = \frac{1}{2} \sum_k (t_{pk} - y_{pk})^2$$

#### Cálculo de las derivadas

Después de obtener las salidas y de haber calculado el error, el
siguiente paso es calcular la derivada del error con respecto de los
pesos. Recordando que $E = \sum_p E_p$ es la suma del error individual
de los patrones, entonces la derivada total es sola la suma de las
derivadas por muestra.

$$\frac{\partial E}{\partial w_{ij}} = \sum_p \frac{\partial E_p}{\partial w_{ij}}.$$

Lo que hace eficiente a la retropropagación (el cálculo de la derivada)
es cómo se descompone la operación y el orden de los pasos.

Conviene calcular de forma separada las derivadas del error con respecto
a los pesos que se conectan a la unidad de salida y para las conexiones
de los nodos ocultos.

La derivada con respecto a las conexiones a las unidades de salida puede
ser escrita como

$$\label{eq:derivative-output}
    \frac{\partial E_p}{\partial w_{jk}} = \frac{\partial E_p}{\partial y_k}\frac{\partial y_k}{\partial a_k} \frac{\partial a_{k}}{\partial w_{jk}}$$

donde $k$ indexa a una unidad de salida y $a_k$ se calcula utilizando la
ecuación
[\[linear\_combination\_aj\]](#linear_combination_aj){reference-type="ref"
reference="linear_combination_aj"}. Conviene primero calcular un valor
$\delta_k$ para cada nodo de salida $k$. Este valor delta también es
conocido como *error de retropropagación*.

$$\begin{aligned}
\delta_k &= \frac{\partial E_p}{\partial y_k}\frac{\partial y_k}{\partial a_k} \\
&= -(t_k - y_k) f'(a_k).\end{aligned}$$

Para el tercer término de
([\[eq:derivative-output\]](#eq:derivative-output){reference-type="ref"
reference="eq:derivative-output"}), como $a_k$ es una suma lineal, es
cero si $i \neq j$, de otra forma $$\begin{aligned}
    \frac{\partial a_k}{\partial w_{jk}} &=  \frac{\partial \sum_i w_{ik}y_i}{\partial w_{jk}}\\
    &= y_j.\end{aligned}$$ Por lo tanto $$\begin{aligned}
    \frac{\partial E_p}{\partial w_{jk}} = \delta_k y_j \end{aligned}$$

El segundo caso corresponde al cálculo de las derivadas con respecto a
los pesos que se conectan a unidades ocultas. El cálculo de la derivada
hasta estos pesos no se obtiene de manera directa, como el caso de las
conexiones a las unidades de salida, por lo que la derivada se obtiene
calculando

$$\begin{aligned}
    \frac{\partial E_p}{ \partial w_{ij}} = \sum_k \frac{\partial E_p}{\partial y_k} \frac{\partial y_k}{\partial a_k} \frac{\partial a_k}{\partial y_j} \frac{\partial y_j}{\partial a_j} \frac{\partial a_j}{\partial w_{ij}}\end{aligned}$$

donde $k$ indexa a todas los nodos a los que se conecta la unidad $j$,
por ahora suponemos que son las $k$ unidades de salida. Simplificando la
expresión, podemos ver que en los primeros dos factores de la suma
estamos calculando los valores delta de las unidades de salida, de ahí
el nombre de error de retropropagación.

$$\begin{aligned}
    \frac{\partial E_p}{ \partial w_{ij}} &= \sum_k \frac{\partial E_p}{\partial a_k} \frac{\partial a_k}{\partial y_j} \frac{\partial y_j}{\partial a_j} \frac{\partial a_j}{\partial w_{ij}}\\
    &= \sum_k \delta_k w_{jk} \frac{\partial y_j}{\partial a_j} \frac{\partial a_j}{\partial w_{ij}} \\
    &= \sum_k \delta_k w_{jk} f'(a_j) \frac{\partial a_j}{\partial w_{ij}} \\
    &= \delta_j y_i\end{aligned}$$

donde $\delta_j = \sum_k \delta_k w_{jk} f'(a_j)$. De manera general
para calcular la derivada de la función de costo con respecto a
cualquiera de los pesos tenemos

$$\frac{\partial E_p}{\partial w_{ij}} = \delta_j y_i$$

donde $\delta_j$ es el error de retroprogación de la unidad $j$, que es
a la unidad a la que llega la arista con el peso $w_{ij}$ y $y_i$ es la
salida de la unidad $i$ que será la entrada del nodo $j$.

\centering
![La retropropagación en una red de tres capas. Las líneas sólidas
muestran el paso del feedforward y las líneas discontinuas muestras la
propagación hacia atrás de los valores
$\delta$.[]{label="fig:backprop"}](/assets/backprop.png)

#### Actualización de los pesos

Después de obtener las derivadas, el siguiente paso es actualizar los
pesos para disminuir el error. Como se dijo al principio, el término de
retropropagación se refiere al método eficiente para calcular las
derivadas $\frac{\partial E}{\partial w}$ y al algoritmo de optimización
que utiliza esas derivadas para ajustar los pesos y reducir el error.

La retropropagación como método de optimización es básicamente el
descenso del gradiente. Sabemos que el gradiente negativo de $E$ apunta
a la dirección en la que $E$ se decrementa más rápido. Para minimizar
$E$, los pesos son ajustados en la dirección del gradiente negativo. La
regla para actualizar los pesos es

$$w_{ij} \leftarrow  w_{ij} - \gamma \frac{\partial E}{\partial w_{ij}}$$

donde la *tasa de aprendizaje* $0 < \gamma$. Hay dos variaciones básicas
para la actualización, el *modo por lotes* y *en línea*.

-   **Aprendizaje por lotes**: En este modo, cada patrón $p$ es evaluado
    para obtener los términos de la derivada
    $\frac{\partial E_p}{\partial w}$; estos se suman para obtener la
    derivada total
    $$\frac{\partial E}{\partial w}  = \sum_p \frac{\partial E_p}{\partial w}$$
    y sólo después de esto se actualizan los pesos. Los pasos son los
    siguientes:

    1.  Por cada patrón $p$ en el conjunto de entrenamiento

        -   Alimentar a la red con el patrón $p$ y hacer la propagación
            hacia adelante para obtener la salidas de la red.

        -   Calcular el error del patrón $E_p$ y retropropagar para
            obtener las derivadas por patrón
            $\frac{\partial E_p}{\partial w}$.

    2.  Sumar todas las derivadas por patrón para obtener la derivada
        total.

    3.  Actualizar los pesos
        $$w \leftarrow w - \gamma \frac{\partial E}{\partial w}$$

    4.  Repetir.

    Cada paso a través de todo el conjunto de entrenamiento se llama
    *época*.

-   **Aprendizaje en línea**: En este modo de aprendizaje, los pesos se
    actualizan después de que se presenta cada patrón. Generalmente, un
    patrón $p$ se elige aleatoriamente y se presenta a la red. La salida
    se compara con el objetivo para ese patrón y los errores son
    propagados hacia atrás para obtener una derivada
    $\frac{\partial E_p}{\partial w}$ para un solo patrón. Los pesos se
    actualizan inmediatamente después, usando el gradiente del error de
    un solo patrón. Los pasos son:

    1.  Elegir aleatoriamente un patrón $p$ del conjunto de
        entrenamiento.

        -   Alimentar a la red con el patrón $p$ y propagar hacia
            adelante para obtener las salidas de la red.

        -   Calcular el error $E_p$ y retropropagar para obtener las
            derivadas $\frac{\partial E_p}{\partial w}$.

    2.  Actualizar los pesos usando la derivada de un solo patrón
        $$w \leftarrow w- \gamma \frac{\partial E_p}{\partial w}$$

    3.  Repetir.

### Retropropagación en una forma matricial

En estructuras de redes prealimentadas con múltiples capas, es
conveniente reescribir el método de retropropagación en una forma tal
que las operaciones se simplifiquen a multiplicaciones de vectores por
matrices, matrices por vectores, matrices por matrices y vectores por
vectores. A continuación definiremos una red con dos capas, una oculta y
una de salida, sin embargo, se puede ver que es posible generalizar para
redes con más capas ocultas. Todas las operaciones que se realizan son
con respecto a una muestra $p$, por lo que dependiendo de cómo se
realice la actualización de los pesos, es como se deben de realizar las
operaciones con las matrices de las derivadas que obtendremos al final.

Consideraremos una red con $n$ unidades de entrada, $k$ unidades ocultas
y $m$ unidades de salida. Hasta ahora la notación utilizada ha evitado
que tratemos cada capa de una red por separado, pero ahora es necesario
mantener un índice de la capa sobre la que se están haciendo los
cálculo, por lo tanto se usa el superíndice $(l)$ para referirnos a la
capa $l$. Los pesos entre la unidad de entrada $i$ y la oculta $j$ se
denotan por $w_{ij}^{(1)}$. El peso entre la unidad oculta $i$ y la de
salida $j$ será $w_{ij}^{(2)}$.

Existen $n \times k$ pesos entre las unidades de entrada y las ocultas y
$k \times m$ entre las ocultas y las de salida. Sea $\mathbf{W_1}$ la
matriz de tamaño $n \times k$ cuyo componente $w_{ij}^{(1)}$ está en la
$i$ ésima fila y la $j$ ésima columna. De manera similar definamos
$\mathbf{W}_2$ como la matriz de $k \times m$ con elementos
$w_{ij}^{(2)}$. El vector de entrada es de tamaño $n$, y lo definimos
como $\mathbf{x} = (x_1, \dots, x_n)$.

\centering
![Arquitectura de la red propuesta para la notación matricial de la
retropropagación.](mat_struct.png)

[\[fig:mat\_mult\_struct\]]{#fig:mat_mult_struct
label="fig:mat_mult_struct"}

Por ahora como función de activación usaremos a la sigmoide, por lo que
la salida $y_j^{(1)}$ de la unidad es

$$y_j^{(1)} = f(\sum_{i}^{n} w_{ij}^{(1)}x_i) = \frac{1}{1+e^{\sum_{i}^{n} w_{ij}^{(1)}x_i}}$$

Las salidas de la capa oculta se pueden obtener aplicando la función de
activación a cada uno de los elementos que resulten de la multiplicación
$\mathbf{x}\mathbf{W}_1$. El vector $\mathbf{y}^{(1)}$ cuyos componentes
son las salidas de las unidades ocultas está dado por

$$\mathbf{y}^{(1)} = f(\mathbf{x}\mathbf{W}_1).$$ Las salidas de las
unidades en la capa de salida se calculan usando el vector
$\mathbf{y}^{(1)} = (y_1^{(1)},\dots, y_k^{(1)})$. La salida de la red
es un vector $m$ dimensional $\mathbf{y}^{(2)}$, donde

$$\mathbf{y}^{(2)} = f(\mathbf{y}^{(1)}\mathbf{W}_2).$$

En el paso de feedforward, el vector $\mathbf{x}$ alimenta a la red. Los
vectores $\mathbf{y}^{(1)}$ y $\mathbf{y}^{(2)}$ son calculados. Además,
en este paso se pueden obtener las derivadas evaluadas de las funciones
de activación, que se pueden escribir en matrices diagonales
$\mathbf{D}_l$. Para nuestra red de dos capas, $\mathbf{D}_2$ contiene
las derivadas evaluadas de las funciones de activación de los nodos de
salida, y $\mathbf{D}_1$ para las unidades ocultas. Por simplicidad,
supusimos que $f$ es la sigmoide y por lo tanto las derivadas son
$f' = y(1-y)$.

$$\mathbf{D}_2 = 
\begin{pmatrix}
    y_1^{(2)}(1-y_1^{(2)}) & 0 & \dots & 0 \\
    0 & y_2^{(2)}(1- y_2^{(2)}) & \dots & 0 \\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & y_m^{(2)}(1-y_m^{(2)}))
\end{pmatrix}$$ y

$$\mathbf{D}_1 = 
\begin{pmatrix}
    y_1^{(1)}(1 - y_1^{(1)}) & 0 & \dots & 0 \\
    0 & y_2^{(1)}(1 - y_2^{(1)}) & \dots & 0 \\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & y_k^{(1)}(1 - y_k^{(1)}))
\end{pmatrix}$$

Para calcular los valores delta de la unidad de salida necesitamos las
derivadas del error con respecto a las salidas. Definimos al vector
$\mathbf{e}$ con las derivadas de las desviaciones cuadráticas como

$$\mathbf{e} = 
\begin{pmatrix}
    - (t_1 - y_1^{(2)}) \\
    - (t_2 - y_2^{(2)}) \\
    \vdots \\
    - (t_m - y_m^{(2)})
\end{pmatrix}$$

Para una unidad de salida
$\delta_i^{(2)} = - (t_i - y_i^{(2)})y_i^{(2)}(1- y_i^{(2)})$. Por lo
tanto el vector $m$ dimensional $\boldsymbol{\delta^{(2)}}$ que contiene
todos los valores delta de la unidad de salida está dado por

$$\boldsymbol{\delta}^{(2)} = \mathbf{D}_2\mathbf{e}.$$

El vector de tamaño $k$ de valores delta en la capa oculta es
$$\boldsymbol{\delta}^{(1)} = \mathbf{D}_1\mathbf{W}_2 \boldsymbol{\delta}^{(2)}.$$

Después de calcular los vectores con los valores delta, es posible
obtener las derivadas del error con respecto a los pesos. Las matrices
con las derivadas del error con respecto los pesos de $\mathbf{W}_2$ y
$\mathbf{W}_1$, son respectivamente:

$$\nabla \mathbf{W}_2 = (\boldsymbol{\delta}^{(2)}\mathbf{y}^{(1)})^T$$
$$\nabla \mathbf{W}_1 = (\boldsymbol{\delta}^{(1)}\mathbf{x})^T$$

$$\nabla\mathbf{W}_2 = 
\begin{pmatrix}
    \frac{\partial E}{w_{11}^{(2)}} & \dots & \frac{\partial E}{w_{1m}^{(2)}} \\
    \vdots & \ddots & \vdots\\
    \frac{\partial E}{w_{k1}^{(2)}} & \dots & \frac{\partial E}{w_{km}^{(2)}}
\end{pmatrix}$$

$$\nabla\mathbf{W}_1 = 
\begin{pmatrix}
    \frac{\partial E}{w_{11}^{(1)}} & \dots & \frac{\partial E}{w_{1k}^{(1)}} \\
    \vdots & \ddots & \vdots\\
    \frac{\partial E}{w_{n1}^{(1)}} & \dots & \frac{\partial E}{w_{nk}^{(1)}}
\end{pmatrix}$$ Es fácil generalizar estas ecuaciones para $l$ capas de
unidades de cómputo. Asumamos que la matriz de conexión entre la capa
$i$ e $i+1$ está denotada por $\mathbf{W}_{i+1}$. El vector
$\mathbf{\delta}^{(l)}$ de la capa de salida es entonces

$$\boldsymbol{\delta}^{(l)} = \mathbf{D}_l\mathbf{e}$$

El vector $\mathbf{\delta}^{(i)}$ hasta la $i$ ésima capa se define
recursivamente

$$\boldsymbol{\delta}^{(i)} = \mathbf{D}_i\mathbf{W}_{i+1} \boldsymbol{\delta}^{(i+1)} \mbox{ para } i = 1, \dots, l - 1$$

o de manera alternativa

$$\boldsymbol{\delta}^{(i)} = \mathbf{D}_i\mathbf{W}_{i+1} \dots \mathbf{W}_{l-1}\mathbf{D}_{l-1}\mathbf{W}_l\mathbf{D}_{l} \mathbf{e}$$

Las correcciones para las matrices de pesos se calculan de la misma
manera que para las dos capas de las unidades de cómputo.
