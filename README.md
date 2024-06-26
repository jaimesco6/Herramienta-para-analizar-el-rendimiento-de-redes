# Herramienta para analizar el rendimiento de redes

## Contexto
Desde el comienzo de la Era Digital, la industria de las telecomunicaciones ha experimentado un crecimiento exponencial. Las comunicaciones móviles, en particular, han evolucionado a través de varias generaciones tecnológicas, desde los primeros sistemas analógicos hasta las modernas redes 5G. Esta evolución no solo ha mejorado la calidad y velocidad de las comunicaciones, sino que también ha transformado la forma en que las personas y las empresas interactúan con la tecnología. La tecnología móvil se ha convertido en una parte integral de la vida cotidiana, impulsando la conectividad global y facilitando una amplia gama de servicios y aplicaciones. A pesar de esto, existe una gran incertidumbre en cuanto a la verdadera magnitud de la mejora que esta tecnología ha aportado a la calidad de servicio experimentada por los usuarios finales. 

Esta herramienta tiene como objetivo principal analizar y comparar el desempeño de las redes 4G  y 5G, tanto SA (Stand Alone) como NSA (Non-Stand Alone).

## Estructura del proyecto 
```
Herramienta
|__ dataframes
|   |__ dataframe con los datos de las pruebas de speedtest realizadas
|__ Notebook
|   |__ Main.ipynb
|__ Funciones
|   |__ processing.py
|   |__ results.py
README.md
```
## Instalación
Para poder utilizar la herramienta hay que realizar una serie de pasos previos, explicados a continuación.
1. Desde la terminal de anaconda, crear un nuevo entorno y activarlo:
   ```
   conda create --name herramienta_speedtest python=3.10
   conda activate herramienta_speedtest
   ```
2. Clonal este repositorio:
   ```
   git clone https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes
   ```
3. Accedemos al directorio donde se encuentra en repositorio:
   ```
   cd Herramienta-para-analizar-el-rendimiento-de-redes/Herramienta
   ```
5. Accedemos al notebook que representa la interfaz de la herramienta:
   ```
   jupyter notebook Main.ipynb
   ```
7. Una vez dentro, simplemente hay que introducir un fichero .csv que contenga datos de pruebas de speedtest realizadas para 5G SA y 4G y otro que contenga medidas de 5G SA. A continuación, se podrá utilizar la herramienta ejecutando las celdas del notebook y ajustando los parámetros de entrada de las funciones.
   
## Preprocesado/procesado de los datos
Como primer paso, la herramienta realiza tanto el preprocesado como el procesado de los datos en crudo, para tener unos valores adecuados. Para ello, debe ejecutarse la función llamada function_processing del fichero processing.py.
Esta función, a su vez, contiene otras funciones que realizan las tareas necesarias. Estas tareas son:

1. **Cálculo de tasa binaria** --> Transformación de los valores de throughput de Bytes a Mbits.
2. **Creación de nuevas variables a partir de las disponibles originalmente:**
   - Dispositivo
   - Banda de frecuencia
   - Banda de frecuencia de la celda secundaria
   - Tecnología
   - GNodeB / eNodeB
3. **Detección de casos anómalos** --> Detección de aquellos datos que han sido tomados de manera defectuosa.
   - CI (Cell Identifier)
   - Banda de frecuencia
4. **Agrupar medidas por geohashes** --> Las medidas se agrupan por geohashes, dependiendo de su ubicación.
5. **Ajustar ubicaciones** --> el usuario debe modificar la localización de las medidas imprecisas.
6. **División de los datos por tecnología**
   - 5G SA --> df_5G_SA
   - 5G NSA --> df_5G_NSA
   - 4G --> df_4G

## Resultados
Una vez se dispone de los datos en un formato adecuado, se va a proceder a realizar el análisis de los resultados, observando diferentes variables. Para hacer esto, se importan todas las funciones definidas en el fichero results.py.

### Ubicar en el mapa las medidas seleccionando el día
Ejecutando la función create_map_from_dataframe se generá un mapa interactivo que muestra la localización de los datos para un día y tecnología determinado.
Esta función toma tres parámetros como entrada:
1. Tecnología que se quiera representar --> df_5G_SA, df_5G_NSA o df_4G.
2. Mes en el que se han tomado las medidas.
3. Dia en el que se han tomado las medidas.
A continuación se muestra un ejemplo de lo explicado.

<p align="center">
  <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/17a94807-15c2-4b52-a836-e86933f82fed" alt="image" width="300" height="350">
</p>

### Representación gráfica de las medidas mediante ventana deslizante de 50 elementos
Para representar de manera gráfica los valores de throughput de las pruebas de speedtest realizadas, hay que ejecutar la función llamada generate_plots. Toma como parámetros de entrada los siguientes:
1. Dataframe de la tecnología
2. Tiempo de muestreo (dependiendo de los datos del usuario)
3. Tamaño de la ventana (a elegir por el usuario)
4. Modo:
   - "Completo" (viene de serie) --> muestra todas las pruebas del dataframe seleccionado. **Aviso:** dependiendo del número de medidas puede tardar en mostrar las gráficas.
   - Parcial: muestra diez gráficas de manera aleatoria.

A continuación se muestra un ejemplo.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/91385876-a314-4e68-9c57-6b1e748afab4" alt="image" width="650" height="450">
</p>

### Agrupar las medidas por geohashes
Se añade una nueva columna que indica el geohash al que pertenece cada una de las pruebas realizadas. La función toma dos parámetros:
1. Dataframe de la tecnología seleccionada.
2. Precisión del geohash, valor numérico entero.
   
### Violinplots por cada geohash
Ejecutando la función llamada setup_interactive_geohash_selection, se despliega un menú interactivo con dos grupos de pestañas. En el primero se puede seleccionar el geohash específico que se quiera analizar, y en el segundo se elige el tipo de estadístico a partir del cual se crean los violinplots. Estos estadísticos son la media, mediana, percentil 10 y percentil 90. También representa en un mapa el geohash seleccionado.
Los parámetros de entrada que hay que introducir en la función son:
1. Dataframe de la tecnología 1.
2. Dataframe de la tecnología 2.
3. Dataframe de la tecnología 3.
4. Tiempo de muestreo.
5. Tamaño de la ventana deslizante.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/2f81a23c-8abf-4d71-bddd-1b23cb6fcff9" alt="image"width="650" height="450">
</p>

### Distribuciones del throughput media por tecnología
La función plot_distribucion_medias representa tres histogramas superpuestos (uno por tecnología) de los valores medios de throughput de las pruebas. La función toma tres parámetros como entrada.
1. df_5G_SA
2. df_5G_NSA
3. df_4G
4. Array de bins adecuado a los datos disponibles
5. Normalizar:
   - "True" --> se normalizan los valores para realizar una comparación justa
   - "False" --> no se normalizan los valores (recomendado cuando hay un numero de medidas muy similar, entre tecnologías)

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/d1c8b4c2-7fb4-429a-ab40-d3131b922018" alt="image"width="650" height="450">
</p>

### Boxplot comparativo de los valores medios de throughput
La función plot_boxplot_medias representa los boxplots de las tres tecnologías. Toma tres parámetros de entrada.
1. Valores de throughput medio de 5G SA.
2. Valores de throughput medio de 5G NSA.
3. Valores de throughput medio de 4G.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/9885d6bc-da31-43e8-b6c3-6c9f71584ff1" alt="image"width="650" height="450">
</p>

### Test ANOVA
Primero se utiliza la función llamada anova, que toma como entradas los dataframes de las tres tecnologías, que representa la tabla anova para ver si hay diferencias significativas entre los valores de throughput de las tres tecnologías. También se ejecuta la función tukey_hsd que representa los resultados de aquellas variables considerablemente diferentes.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/fb8fb9ee-f3ee-46f1-b0c5-72ac9c62a311" alt="image"width="250" height="50">
</p>
<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/2e59f5f6-4418-40e7-a806-223e5f94d9ec" alt="image"width="650" height="450">
</p>

### Boxplots por banda de frecuencia
Para cada una de las tecnologías, se va representar un boxplot por cada una de las bandas de frecuencia utilizadas, para cada tecnología. La función plot_boxplots_por_banda toma como parámetros los dataframes de las tecnologías. Se representa una imagen como ejemplo.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/8fd17137-1d41-41b0-9d33-0048c7b74213" alt="image"width="675" height="400">
</p>

### Bandas de frecuencia más utilizadas por frecuencia
La función encargada de representar las medidas tomadas en cada una de las bandas de frecuencia, plot_separate_frequency_bands, toma como parámetros de entradalos tres dataframes.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/fc3d22bd-c888-48b3-903c-1fd749acc030" alt="image"width="675" height="400">
</p>

### Análisis de latencia
La función latency_avg representa un histograma con los valores de latencia media de las pruebas de velocidad. Toma como parámetro de entrada el dataframe que se quiera analizar.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/94328306-eb09-410a-b826-1355ad362d1d" alt="image"width="650" height="450">
</p>

### Análisis de RSRP
La función analize_rsrp toma como parámetro de entrada el dataframe y muestra un histograma de los valores de RSRP para dicha tecnología. En la leyenda representa el porcentaje de valores con una RSRP excelente, buena, aceptable y pobre tanto al comienzo como al final de la prueba.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/2d512dbd-fc9c-4787-8fb3-52864215916c" alt="image"width="775" height="350">
</p>

### Análisis de RSRQ
La función analize_rsrq toma como parámetro de entrada el dataframe y muestra un histograma de los valores de RSRQ para dicha tecnología. En la leyenda representa el porcentaje de valores con una RSRP excelente, buena, aceptable y pobre tanto al comienzo como al final de la prueba.

<p align="center">
   <img src="https://github.com/jaimesco6/Herramienta-para-analizar-el-rendimiento-de-redes/assets/167304557/6272f25e-2c62-4a4c-9750-24495b3d310c" alt="image"width="775" height="350">
</p>
