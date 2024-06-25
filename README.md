# Herramienta para analizar el rendimiento de redes

## Contexto
Desde el comienzo de la Era Digital, la industria de las telecomunicaciones ha experimentado un crecimiento exponencial. Las comunicaciones móviles, en particular, han evolucionado a través de varias generaciones tecnológicas, desde los primeros sistemas analógicos hasta las modernas redes 5G. Esta evolución no solo ha mejorado la calidad y velocidad de las comunicaciones, sino que también ha transformado la forma en que las personas y las empresas interactúan con la tecnología. La tecnología móvil se ha convertido en una parte integral de la vida cotidiana, impulsando la conectividad global y facilitando una amplia gama de servicios y aplicaciones. A pesar de esto, existe una gran incertidumbre en cuanto a la verdadera magnitud de la mejora que esta tecnología ha aportado a la calidad de servicio experimentada por los usuarios finales. 

Esta herramienta tiene como objetivo principal analizar y comparar el desempeño de las redes 4G  y 5G, tanto SA (Stand Alone) como NSA (Non-Stand Alone).

## Estructura del proyecto 
```
Herramienta
|__ Datos
|   |__ dataframe con los datos de las pruebas de speedtest realizadas
|__ Notebook
|   |__ Main.ipynb
|__ Funciones
|   |__ processing.py
|   |__ results.py
README.md
```
### Instalación

## 1. Preprocesado/procesado de los datos
Como primer paso, se realizará tanto el preprocesado como el procesado de los datos en crudo. Para ello, debe ejecutarse la función llamada function_processing del fichero processing.py.
Esta función, a su vez, contiene otras funciones que realizan las tareas necesarias. Estas tareas son:

1. Cálculo de tasa binaria --> Transformación de los valores de throughput de Bytes a Mbits.
2. Creación de nuevas variables a partir de las disponibles originalmente:
   - Dispositivo
   - Banda de frecuencia
   - Banda de frecuencia de la celda secundaria
   - Tecnología
   - GNodeB / eNodeB
3. Detección de casos anómalos --> Detección de aquellos datos que han sido tomados de manera defectuosa.
   - CI (Cell Identifier)
   - Banda de frecuencia
4. Agrupar medidas por geohashes --> Las medidas se agrupan por geohashes, dependiendo de su ubicación.
5. Ajustar ubicaciones --> el usuario debe modificar la localización de las medidas imprecisas.
6.  División de los datos por tecnología
   - 5G SA
   - 5G NSA
   - 4G
