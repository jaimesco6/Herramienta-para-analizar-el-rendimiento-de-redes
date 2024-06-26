import pandas as pd
import ast
import numpy as np
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import geohash2 as gh2
import pygeohash as pgh
from IPython.display import display
import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sn
import re
import random

sampling_time = 0.05

def process_dataframes_separated(df_SA_4G_processed, df_NSA_processed):
    """
    Procesa dos DataFrames especificados: df_SA_4G_processed y df_NSA_processed, aplicando una serie de
    transformaciones y filtros según la tecnología ('5G SA', '5G NSA', '4G'), devolviendo tres DataFrames
    separados para cada grupo de tecnología.

    Args:
    df_SA_4G_processed (pd.DataFrame): DataFrame con datos de tecnología 'SA 4G'.
    df_NSA_processed (pd.DataFrame): DataFrame con datos de tecnología 'NSA'.

    Returns:
    tuple: Retorna tres DataFrames separados: df_5G_SA, df_5G_NSA, df_4G.
    """
    # Procesamiento para '5G SA'
    df_5G_SA = df_SA_4G_processed[df_SA_4G_processed['technology'] == '5G SA'].copy()
    df_5G_SA.dropna(subset=['ci_start', 'ci_end'], how='any', inplace=True)
    df_5G_SA['ci_start'].fillna(df_5G_SA['ci_end'], inplace=True)
    df_5G_SA['ci_end'].fillna(df_5G_SA['ci_start'], inplace=True)
    df_5G_SA['gnodeb_start'] = (df_5G_SA['ci_start'] // 16384).astype(int).astype(str) + '-' + (df_5G_SA['ci_start'] % 16384).astype(int).astype(str)
    df_5G_SA['gnodeb_end'] = (df_5G_SA['ci_end'] // 16384).astype(int).astype(str) + '-' + (df_5G_SA['ci_end'] % 16384).astype(int).astype(str)

    # Procesamiento para '5G NSA'
    df_5G_NSA = df_NSA_processed[df_NSA_processed['technology'] == '5G NSA'].copy()
    df_5G_NSA.dropna(subset=['ci_start', 'ci_end'], how='any', inplace=True)
    df_5G_NSA['ci_start'].fillna(df_5G_NSA['ci_end'], inplace=True)
    df_5G_NSA['ci_end'].fillna(df_5G_NSA['ci_start'], inplace=True)
    df_5G_NSA['gnodeb_start'] = (df_5G_NSA['ci_start'] // 256).astype(int).astype(str) + '-' + (df_5G_NSA['ci_start'] % 256).astype(int).astype(str)
    df_5G_NSA['gnodeb_end'] = (df_5G_NSA['ci_end'] // 256).astype(int).astype(str) + '-' + (df_5G_NSA['ci_end'] % 256).astype(int).astype(str)

    # Procesamiento adicional para '4G'
    df_4G = df_SA_4G_processed[df_SA_4G_processed['technology'] == '4G'].copy()
    df_4G.dropna(subset=['ci_start', 'ci_end'], how='any', inplace=True)
    df_4G['ci_start'].fillna(df_4G['ci_end'], inplace=True)
    df_4G['ci_end'].fillna(df_4G['ci_start'], inplace=True)
    df_4G['gnodeb_start'] = (df_4G['ci_start'] // 256).astype(int).astype(str) + '-' + (df_4G['ci_start'] % 256).astype(int).astype(str)
    df_4G['gnodeb_end'] = (df_4G['ci_end'] // 256).astype(int).astype(str) + '-' + (df_4G['ci_end'] % 256).astype(int).astype(str)

    return df_5G_SA, df_5G_NSA, df_4G


def create_map_from_dataframe(df, month, day):
    """
    Crea un mapa utilizando folium para visualizar marcadores de las ubicaciones basadas en un DataFrame
    filtrado por un mes y día específicos.

    Args:
    df (pd.DataFrame): DataFrame con datos que incluyen 'latitude', 'longitude', 'day' y 'month'.
    month (int): Mes específico para filtrar dentro del DataFrame.
    day (int): Día específico para filtrar dentro del DataFrame.

    Returns:
    folium.Map: Un objeto de mapa de folium con marcadores agregados.
    """
    # Filtrar el DataFrame por el mes y día especificados
    df_filtered = df[(df['month'] == month) & (df['day'] == day)]

    # Crear un mapa centrado en la media de las ubicaciones del mes y día especificados
    if not df_filtered.empty:
        mapa = folium.Map(location=[df_filtered['latitude'].mean(), df_filtered['longitude'].mean()], zoom_start=10)
        cluster = MarkerCluster().add_to(mapa)

        # Agregar marcadores al cluster
        for index, row in df_filtered.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']], 
                popup=f"Month: {row['month']}<br>Day: {row['day']}"
            ).add_to(cluster)

        return mapa
    else:
        return "No hay mediciones para el mes y día seleccionados"
    
    
import random

def generate_plots(df, sampling_time, window_size, mode='completo'):
    """
    Genera gráficos de los valores de throughput para las filas seleccionadas en el DataFrame especificado,
    utilizando un tamaño de ventana, tiempo de muestreo, y un modo que puede ser 'completo' o 'parcial'.

    Args:
    df (pd.DataFrame): DataFrame que contiene las columnas necesarias, incluyendo 'dl_th_list'.
    sampling_time (float): El intervalo de tiempo de muestreo.
    window_size (int): El tamaño de la ventana para la media móvil.
    mode (str): Modo de visualización de los datos, 'completo' para todos los datos o 'parcial' para una muestra aleatoria de 10 filas.

    """
    # Seleccionar filas según el modo
    if mode == 'parcial':
        if len(df) > 10:
            df = df.sample(n=10, random_state=1)  # Usar un estado aleatorio fijo para reproducibilidad
    elif mode != 'completo':
        raise ValueError("Mode must be 'completo' or 'parcial'")
    
    for index, row in df.iterrows():
        plt.figure(figsize=(10, 6))
        
        # Convertir la lista a un arreglo numpy
        dl_th_array = np.array(row['dl_th_list'])
        
        # Ventana deslizante (tamaño especificado por window_size)
        medicion_with_zeros = np.concatenate((np.zeros(window_size - 1), dl_th_array / sampling_time))
        transformed_values_2 = np.convolve(medicion_with_zeros, np.ones(window_size)/window_size, mode='valid')
        
        # Valores en crudo
        start_index = window_size // 2  # Ajustar el inicio para cuadrar con la ventana deslizante
        medicion = dl_th_array / sampling_time
        medicion_padded = np.pad(medicion, (start_index, 0), mode='constant', constant_values=(0,))
        
        # Encontrar el índice donde terminan los ceros al principio
        first_nonzero_index = np.argmax(medicion_padded > 0)
        
        # Calcular el rango para recortar los datos
        end_index = len(transformed_values_2)
        
        # Crear el array de tiempo
        num_points = max(len(transformed_values_2), len(medicion_padded))
        time_array = np.arange(0, num_points * sampling_time, sampling_time)[:num_points]
        
        # Plotear los datos recortados
        plt.plot(time_array[first_nonzero_index:end_index], 
                 transformed_values_2[first_nonzero_index:end_index], 
                 label='Ventana deslizante (tamaño {})'.format(window_size))
        
        plt.plot(time_array[first_nonzero_index:end_index], 
                 medicion_padded[first_nonzero_index:end_index], 
                 label='Valores en crudo', color="#DDDDDD")

        plt.xlabel('Tiempo (s)') 
        plt.ylabel('Valor (Mbps)')
        plt.title('Throughput')
        plt.legend()
        plt.grid(False)  # Desactivar las cuadrículas
        
        # Calcular la media de la ventana deslizante
        mean_sliding_window = np.mean(transformed_values_2[first_nonzero_index:end_index])
        
        # Agregar el valor medio a la leyenda
        plt.text(0.5, 0.95, f'Media ventana deslizante: {mean_sliding_window:.2f} Mbps', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.show()

        
def add_geohash_column(df, precision=6):
    """
    Añade una columna de geohash al DataFrame dado, basada en las columnas 'latitude' y 'longitude' utilizando la biblioteca geohash2.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos, incluidas las columnas 'latitude' y 'longitude'.
    precision (int): Precisión del geohash, que determina la granularidad de la agrupación geográfica.

    Returns:
    pd.DataFrame: El mismo DataFrame con una nueva columna 'geohash' agregada.
    """
    # Crear geohashes a partir de latitud y longitud y añadir como nueva columna
    df['geohash'] = df.apply(lambda row: gh2.encode(row['latitude'], row['longitude'], precision=precision), axis=1)
    
    return df

def visualize_geohashes(df, geohash_col='geohash'):
    """
    Visualiza geohashes en un mapa interactivo utilizando la biblioteca folium, mostrando el nombre del geohash y
    el número de registros para cada geohash en un popup al hacer clic en los rectángulos.

    Args:
    df (pd.DataFrame): DataFrame que contiene una columna de geohashes.
    geohash_col (str): Nombre de la columna que contiene los geohashes.

    Returns:
    folium.Map: Un mapa de folium con las cajas de geohash dibujadas, con popups que muestran el nombre del geohash y el número de registros.
    """
    # Crear un mapa base centrado en un punto neutro inicial
    map = folium.Map(location=[0, 0], zoom_start=2)
    
    # Contar las ocurrencias de cada geohash
    geohash_counts = df[geohash_col].value_counts()
    
    # Listas para almacenar las coordenadas decodificadas
    latitudes = []
    longitudes = []
    
    # Añadir las cajas de geohash al mapa
    for geohash in df[geohash_col].unique():
        try:
            # Decodificar geohash
            box = pgh.decode_exactly(geohash)
            bounds = [
                (box[0] - box[2], box[1] - box[3]),  # Southwest corner
                (box[0] + box[2], box[1] + box[3])   # Northeast corner
            ]
            # Crear un rectángulo para el geohash con un popup
            folium.Rectangle(
                bounds=bounds,
                color='#0078FF',
                fill=True,
                fill_opacity=0.4,
                popup=f'Geohash: {geohash}\nMedidas: {geohash_counts.get(geohash, 0)}'
            ).add_to(map)
            latitudes.append(box[0])
            longitudes.append(box[1])
        except ValueError:
            print(f"Error decoding geohash {geohash}")
    
    # Calcular la ubicación central del mapa basada en los geohashes decodificados
    if latitudes and longitudes:
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        map.location = [center_lat, center_lon]
        map.zoom_start = 5  # Ajustar según sea necesario
    
    return map

def calculate_statistic(values, estadistico):
    if estadistico == 'mediana':
        return pd.Series(values).median()
    elif estadistico == 'media':
        return pd.Series(values).mean()
    elif estadistico == 'percentil_10':
        return pd.Series(values).quantile(0.10)
    elif estadistico == 'percentil_90':
        return pd.Series(values).quantile(0.90)
    else:
        raise ValueError("El parámetro 'estadistico' debe ser 'media', 'mediana', 'percentil_10' o 'percentil_90'.")

def get_values(df, estadistico, sampling_time):
    # Crear una copia del DataFrame para evitar SettingWithCopyWarning
    df = df.copy()
    
    # Calcular la estadística para cada fila
    df['stat_value'] = df['dl_th_list'].apply(lambda x: calculate_statistic([value / sampling_time for value in x], estadistico))
    
    # Expandir y calcular las estadísticas acumuladas
    if estadistico in ['mediana', 'media', 'percentil_10', 'percentil_90']:
        return df['stat_value'].expanding().mean().tolist()
    else:
        raise ValueError("El parámetro 'estadistico' debe ser 'media', 'mediana', 'percentil_10' o 'percentil_90'.")

def plot_data(geohash, df_SA_geohash, df_NSA_geohash, df_4G_geohash, sampling_time, window_size, estadistico='media'):
    """
    Genera un violinplot y un mapa para un geohash específico a partir de los DataFrames dados.

    Args:
    geohash (str): El geohash para el cual generar los plots.
    df_SA_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para SA.
    df_NSA_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para NSA.
    df_4G_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para 4G.
    sampling_time (float): El intervalo de tiempo de muestreo.
    window_size (int): El tamaño de la ventana para la media móvil.
    estadistico (str): Indica si se deben calcular 'media', 'mediana', 'percentil_10' o 'percentil_90'.
    """
    df_SA_filtered = df_SA_geohash[df_SA_geohash['geohash'] == geohash].copy()
    df_NSA_filtered = df_NSA_geohash[df_NSA_geohash['geohash'] == geohash].copy()
    df_4G_filtered = df_4G_geohash[df_4G_geohash['geohash'] == geohash].copy()

    values_SA = get_values(df_SA_filtered, estadistico, sampling_time)
    values_NSA = get_values(df_NSA_filtered, estadistico, sampling_time)
    values_4G = get_values(df_4G_filtered, estadistico, sampling_time)

    data = {
        'SA': values_SA,
        'NSA': values_NSA,
        '4G': values_4G
    }
    df_plot = pd.DataFrame({k: pd.Series(v, dtype='float64') for k, v in data.items()}).melt(var_name='Network Technology', value_name='Throughput (Mbps)')

    plt.figure(figsize=(10, 6))
    ax = sn.violinplot(x='Network Technology', y='Throughput (Mbps)', data=df_plot)

    counts = [len(values_SA), len(values_NSA), len(values_4G)]
    for i, count in enumerate(counts):
        ax.text(i, max(df_plot['Throughput (Mbps)']) + 0.5, f'{count} medidas', horizontalalignment='center', size='medium', color='black', weight='semibold')

    plt.xlabel('Tecnología')
    if estadistico == 'mediana':
        plt.ylabel('Mediana de Throughput (Mbps)')
    elif estadistico == 'media':
        plt.ylabel('Media de Throughput (Mbps)')
    elif estadistico == 'percentil_10':
        plt.ylabel('Percentil 10 de Throughput (Mbps)')
    elif estadistico == 'percentil_90':
        plt.ylabel('Percentil 90 de Throughput (Mbps)')
    plt.title(f'Representación de {estadistico.capitalize()} para el Geohash {geohash}')
    plt.grid(True)
    plt.show()

    box = pgh.decode_exactly(geohash)
    folium_map = folium.Map(location=[box[0], box[1]], zoom_start=12)
    folium.Rectangle(
        bounds=[(box[0] - box[2], box[1] - box[3]), (box[0] + box[2], box[1] + box[3])],
        color='blue', fill=True, fill_opacity=0.5
    ).add_to(folium_map)
    display(folium_map)

    
def setup_interactive_geohash_selection(df_SA_geohash, df_NSA_geohash, df_4G_geohash, sampling_time, window_size):
    """
    Configura un widget interactivo para seleccionar geohashes y visualizar los datos.

    Args:
    df_SA_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para SA.
    df_NSA_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para NSA.
    df_4G_geohash (pd.DataFrame): DataFrame que contiene geohashes y dl_th_list para 4G.
    sampling_time (float): El intervalo de tiempo de muestreo.
    window_size (int): El tamaño de la ventana para la media móvil.
    """
    unique_geohashes = pd.concat([df_SA_geohash['geohash'], df_NSA_geohash['geohash'], df_4G_geohash['geohash']]).unique()
    geohash_selector = widgets.Select(
        options=unique_geohashes,
        value=unique_geohashes[0],
        description='Geohash:',
        disabled=False
    )
    estadistico_selector = widgets.ToggleButtons(
        options=[('Media', 'media'), ('Mediana', 'mediana'), ('Percentil 10', 'percentil_10'), ('Percentil 90', 'percentil_90')],
        value='media',
        description='Estadístico:'
    )
    widgets.interact(lambda geohash, estadistico: plot_data(geohash, df_SA_geohash, df_NSA_geohash, df_4G_geohash, sampling_time, window_size, estadistico), 
             geohash=geohash_selector, estadistico=estadistico_selector)



palette = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)]

def plot_distribucion_medias(df_5G_SA, df_5G_NSA, bins_array, df_4G=None, normalize=False):
    """
    Genera un gráfico comparativo de la distribución de las medias de throughput para tres DataFrames.

    Args:
    df_5G_SA (pd.DataFrame): DataFrame que contiene los datos de 5G SA, incluyendo la columna 'mean'.
    df_5G_NSA (pd.DataFrame): DataFrame que contiene los datos de 5G NSA, incluyendo la columna 'mean'.
    bins_array (array): Array con los límites de los bins.
    df_4G (pd.DataFrame, optional): DataFrame que contiene los datos de 4G, incluyendo la columna 'mean'. Default es None.
    normalize (bool, optional): Si se debe normalizar los histogramas. Default es False.
    """
    plt.figure(figsize=(12, 8))
    
    plt.hist(df_5G_SA['mean'], bins=bins_array, alpha=0.5, label='5G SA', edgecolor='black', density=normalize)
    plt.hist(df_5G_NSA['mean'], bins=bins_array, alpha=0.5, label='5G NSA', edgecolor='black', density=normalize)
    
    if df_4G is not None:
        plt.hist(df_4G['mean'], bins=bins_array, alpha=0.5, label='4G', edgecolor='black', density=normalize)
    
    plt.xlabel('Media de Throughput (Mbps)')
    plt.ylabel('Frecuencia' if not normalize else 'Densidad')
    plt.title('Distribución Comparativa de las Medias de Throughput')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_boxplot_medias(medias_5G_SA, medias_5G_NSA, medias_4G):
    """
    Genera un boxplot comparativo de las medias calculadas para tres DataFrames.

    Args:
    medias_5G_SA (list): Lista de medias calculadas para el DataFrame df_5G_SA.
    medias_5G_NSA (list): Lista de medias calculadas para el DataFrame df_5G_NSA.
    medias_4G (list, optional): Lista de medias calculadas para el DataFrame df_4G. Default es None.
    """
    data = [medias_5G_SA, medias_5G_NSA, medias_4G]
    labels = ['5G SA', '5G NSA', '4G']
    
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Media de Throughput (Mbps)')
    plt.title('Boxplot Comparativo de las Medias de Throughput')
    plt.grid(True)
    plt.show()

def calcular_medias_por_banda(df, band_column):
    """
    Calcula las medias de throughput agrupadas por bandas de frecuencia.

    Args:
    df (pd.DataFrame): DataFrame que contiene las columnas necesarias.
    band_column (str): Nombre de la columna que contiene las bandas de frecuencia.

    Returns:
    dict: Un diccionario donde las claves son las bandas de frecuencia y los valores son listas de medias.
    """
    bandas = df[band_column].unique()
    medias_por_banda = {banda: [] for banda in bandas}
    
    for banda in bandas:
        df_banda = df[df[band_column] == banda]
        medias = df_banda['mean'].tolist() 
        medias_por_banda[banda].extend(medias)
    
    return medias_por_banda

def plot_boxplots_por_banda(df_5G_SA, df_5G_NSA, df_4G):
    """
    Genera boxplots para cada una de las bandas de frecuencia de los DataFrames dados.

    Args:
    df_5G_SA (pd.DataFrame): DataFrame de 5G SA.
    df_5G_NSA (pd.DataFrame): DataFrame de 5G NSA.
    df_4G (pd.DataFrame): DataFrame de 4G.
    """
    medias_5G_SA = calcular_medias_por_banda(df_5G_SA, 'frequency_band_start')
    medias_5G_NSA = calcular_medias_por_banda(df_5G_NSA, 'secondary_frequency_band_start')
    medias_4G = calcular_medias_por_banda(df_4G, 'frequency_band_start')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # 5G SA
    axes[0].boxplot([medias for medias in medias_5G_SA.values()], labels=medias_5G_SA.keys())
    for i, (banda, medias) in enumerate(medias_5G_SA.items()):
        axes[0].text(i + 1, max(medias), f'n={len(medias)}', ha='center', va='bottom')
    axes[0].set_title('5G SA - Distribución de Throughput por Banda de Frecuencia')
    axes[0].set_xlabel('Bandas de Frecuencia')
    axes[0].set_ylabel('Media de Throughput (Mbps)')
    axes[0].grid(True)

    # 5G NSA
    axes[1].boxplot([medias for medias in medias_5G_NSA.values()], labels=medias_5G_NSA.keys())
    for i, (banda, medias) in enumerate(medias_5G_NSA.items()):
        axes[1].text(i + 1, max(medias), f'n={len(medias)}', ha='center', va='bottom')
    axes[1].set_title('5G NSA - Distribución de Throughput por Banda de Frecuencia')
    axes[1].set_xlabel('Bandas de Frecuencia')
    axes[1].set_ylabel('Media de Throughput (Mbps)')
    axes[1].grid(True)

    # 4G
    axes[2].boxplot([medias for medias in medias_4G.values()], labels=medias_4G.keys())
    for i, (banda, medias) in enumerate(medias_4G.items()):
        axes[2].text(i + 1, max(medias), f'n={len(medias)}', ha='center', va='bottom')
    axes[2].set_title('4G - Distribución de Throughput por Banda de Frecuencia')
    axes[2].set_xlabel('Bandas de Frecuencia')
    axes[2].set_ylabel('Media de Throughput (Mbps)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_separate_frequency_bands(df_4G, df_5G_SA, df_5G_NSA):
    # Orden fijo de las etiquetas
    desired_order = [
        'LTE-SDK-700', 'LTE-SDK-800', 'LTE-SDK-900', 'LTE-SDK-1800',
        'LTE-SDK-2100', 'LTE-SDK-2600', 'NR-SDK-700', 'NR-SDK-2100', 'NR-SDK-3500'
    ]

    # Obtener los valores de frecuencia
    frequency_5G_SA = df_5G_SA['frequency_band_start'].value_counts()
    frequency_5G_NSA = df_5G_NSA['secondary_frequency_band_start'].value_counts()
    frequency_4G = df_4G['frequency_band_start'].value_counts()

    # Crear listas de valores alineados para cada DataFrame
    aligned_5G_SA = [frequency_5G_SA.get(label, 0) for label in desired_order]
    aligned_5G_NSA = [frequency_5G_NSA.get(label, 0) for label in desired_order]
    aligned_4G = [frequency_4G.get(label, 0) for label in desired_order]

    # Crear un mapa de colores para cada banda de frecuencia
    colors = list(mcolors.TABLEAU_COLORS.keys())[:len(desired_order)]
    color_map = {label: colors[i] for i, label in enumerate(desired_order)}

    # Crear el gráfico separado para cada tecnología
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Graficar 5G SA
    bars = axs[0].bar(desired_order, aligned_5G_SA, color=[color_map[label] for label in desired_order])
    axs[0].set_title('5G SA')
    axs[0].set_xlabel('Bandas de frecuencia')
    axs[0].set_ylabel('Counts')
    axs[0].xaxis.set_major_locator(ticker.FixedLocator(range(len(desired_order))))
    axs[0].xaxis.set_major_formatter(ticker.FixedFormatter(desired_order))
    axs[0].tick_params(axis='x', rotation=45)
    # Añadir etiquetas de conteo en cada barra
    for bar, value in zip(bars, aligned_5G_SA):
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, int(value), ha='center', va='bottom')

    # Graficar 5G NSA
    bars = axs[1].bar(desired_order, aligned_5G_NSA, color=[color_map[label] for label in desired_order])
    axs[1].set_title('5G NSA')
    axs[1].set_xlabel('Bandas de frecuencia')
    axs[1].set_ylabel('Medidas')
    axs[1].xaxis.set_major_locator(ticker.FixedLocator(range(len(desired_order))))
    axs[1].xaxis.set_major_formatter(ticker.FixedFormatter(desired_order))
    axs[1].tick_params(axis='x', rotation=45)
    # Añadir etiquetas de conteo en cada barra
    for bar, value in zip(bars, aligned_5G_NSA):
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, int(value), ha='center', va='bottom')

    # Graficar 4G
    bars = axs[2].bar(desired_order, aligned_4G, color=[color_map[label] for label in desired_order])
    axs[2].set_title('4G')
    axs[2].set_xlabel('Bandas de frecuencia')
    axs[2].set_ylabel('Medidas')
    axs[2].xaxis.set_major_locator(ticker.FixedLocator(range(len(desired_order))))
    axs[2].xaxis.set_major_formatter(ticker.FixedFormatter(desired_order))
    axs[2].tick_params(axis='x', rotation=45)
    # Añadir etiquetas de conteo en cada barra
    for bar, value in zip(bars, aligned_4G):
        yval = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2, yval, int(value), ha='center', va='bottom')

    # Ajustar el diseño
    plt.tight_layout()
    plt.show()
    
import statsmodels.api as sm
from statsmodels.formula.api import ols

def anova(df_4G, df_5G_SA, df_5G_NSA):
    # Seleccionar las columnas relevantes para ANOVA
    df_5G_SA_ANOVA = df_5G_SA[['mean', 'technology']]
    df_5G_NSA_ANOVA = df_5G_NSA[['mean', 'technology']]
    df_4G_ANOVA = df_4G[['mean', 'technology']]

    # Combinar los DataFrames
    df_ANOVA = pd.concat([df_5G_SA_ANOVA, df_5G_NSA_ANOVA, df_4G_ANOVA])

    # Realizar ANOVA
    modelo = ols('mean ~ C(technology)', data=df_ANOVA).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    
    return anova_table

from statsmodels.stats.multicomp import pairwise_tukeyhsd

def tukey_hsd(df_4G, df_5G_SA, df_5G_NSA):
    # Seleccionar las columnas relevantes para ANOVA
    df_5G_SA_ANOVA = df_5G_SA[['mean', 'technology']]
    df_5G_NSA_ANOVA = df_5G_NSA[['mean', 'technology']]
    df_4G_ANOVA = df_4G[['mean', 'technology']]

    # Combinar los DataFrames
    df_ANOVA = pd.concat([df_5G_SA_ANOVA, df_5G_NSA_ANOVA, df_4G_ANOVA])
    
    # Realizar la prueba de Tukey HSD
    tukey = pairwise_tukeyhsd(endog=df_ANOVA['mean'], groups=df_ANOVA['technology'], alpha=0.05)

    # Mostrar el resumen de la prueba de Tukey
    print(tukey)

    # Visualizar el resumen de la prueba de Tukey
    fig = tukey.plot_simultaneous()
    plt.title('Resultados de la prueba de Tukey HSD')
    plt.show()
    
    return tukey

def latency_avg(df):
    """
    Extrae y calcula la media de los valores 'avg' en la columna 'latency' de un DataFrame.
    También muestra un histograma de los valores 'avg'.

    Args:
    df (pd.DataFrame): DataFrame que contiene la columna 'latency' con los valores de latencia.

    Returns:
    float: La media de los valores 'avg'. Si no se encuentran valores 'avg', devuelve None.
    """

    def extract_avg(latency_str):
        if isinstance(latency_str, str):  # Asegurarse de que sea una cadena de texto
            match = re.search(r'avg=(\d+\.\d+)', latency_str)
            if match:
                return float(match.group(1))
        return None

    # Aplicar la función para extraer los valores de avg
    df_latency = df['latency'].apply(extract_avg)

    # Calcular la media de los valores avg
    avg_mean = df_latency.mean()

    # Crear el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(df_latency.dropna(), bins=20, edgecolor='black')
    plt.title('Histograma de latencia (avg)')
    plt.xlabel('Latencia (ms)')
    plt.ylabel('Frecuencia')
    plt.show()

    return avg_mean

def analize_rsrp(df):
    """
    Analiza los valores de RSRP y genera histogramas para las columnas 'rsrp_start' y 'rsrp_end'.

    Args:
    df (pd.DataFrame): DataFrame que contiene las columnas de RSRP.

    Returns:
    None
    """
    
    column_start = 'rsrp_start'
    column_end = 'rsrp_end'

    # Definir los rangos para rsrp_start
    ranges_start = {
        'Excelente': df[column_start] >= -80,
        'Buena': (df[column_start] < -80) & (df[column_start] > -90),
        'Aceptable': (df[column_start] <= -90) & (df[column_start] > -100),
        'Pobre': df[column_start] <= -100
    }

    # Calcular los porcentajes para rsrp_start
    total_count_start = len(df)
    percentages_start = {label: (condition.sum() / total_count_start) * 100 for label, condition in ranges_start.items()}

    # Definir los rangos para rsrp_end
    ranges_end = {
        'Excelente': df[column_end] >= -80,
        'Buena': (df[column_end] < -80) & (df[column_end] > -90),
        'Aceptable': (df[column_end] <= -90) & (df[column_end] > -100),
        'Pobre': df[column_end] <= -100
    }

    # Calcular los porcentajes para rsrp_end
    total_count_end = len(df)
    percentages_end = {label: (condition.sum() / total_count_end) * 100 for label, condition in ranges_end.items()}

    # Crear el gráfico con dos subgráficos
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Histograma para rsrp_start
    axes[0].hist(df[column_start], bins=20, edgecolor='black')
    axes[0].set_title('Histograma de RSRP al comienzo de la prueba')
    axes[0].set_xlabel('RSRP al comienzo de la prueba (dBm)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True)

    # Mostrar porcentajes en el primer gráfico
    for label, percentage in percentages_start.items():
        axes[0].annotate(f'{label}: {percentage:.2f}%', xy=(0.75, 0.9 - list(percentages_start.keys()).index(label) * 0.05), xycoords='axes fraction')

    # Histograma para rsrp_end
    axes[1].hist(df[column_end], bins=20, edgecolor='black')
    axes[1].set_title('Histograma de RSRP al finalizar la prueba')
    axes[1].set_xlabel('RSRP al finalizar la prueba (dBm)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True)

    # Mostrar porcentajes en el segundo gráfico
    for label, percentage in percentages_end.items():
        axes[1].annotate(f'{label}: {percentage:.2f}%', xy=(0.75, 0.9 - list(percentages_end.keys()).index(label) * 0.05), xycoords='axes fraction')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
    
def analize_rsrq(df):
    """
    Analiza los valores de RSRQ y genera histogramas para las columnas 'rsrq_start' y 'rsrq_end'.

    Args:
    df (pd.DataFrame): DataFrame que contiene las columnas de RSRQ.

    Returns:
    None
    """
    
    column_start = 'rsrq_start'
    column_end = 'rsrq_end'

    # Verificar si el DataFrame contiene las columnas necesarias
    if column_start not in df.columns or column_end not in df.columns:
        print("No se disponen de datos para esta tecnología")
        return

    # Definir los rangos para rsrq_start
    ranges_start = {
        'Excelente': df[column_start] >= -10,
        'Buena': (df[column_start] < -10) & (df[column_start] > -15),
        'Aceptable': (df[column_start] <= -15) & (df[column_start] > -20),
        'Pobre': df[column_start] <= -20
    }

    # Calcular los porcentajes para rsrq_start
    total_count_start = len(df)
    percentages_start = {label: (condition.sum() / total_count_start) * 100 for label, condition in ranges_start.items()}

    # Definir los rangos para rsrq_end
    ranges_end = {
        'Excelente': df[column_end] >= -10,
        'Buena': (df[column_end] < -10) & (df[column_end] > -15),
        'Aceptable': (df[column_end] <= -15) & (df[column_end] > -20),
        'Pobre': df[column_end] <= -20
    }

    # Calcular los porcentajes para rsrq_end
    total_count_end = len(df)
    percentages_end = {label: (condition.sum() / total_count_end) * 100 for label, condition in ranges_end.items()}

    # Crear el gráfico con dos subgráficos
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Histograma para rsrq_start
    axes[0].hist(df[column_start], bins=20, edgecolor='black')
    axes[0].set_title('Histograma de RSRQ Start')
    axes[0].set_xlabel('RSRQ Start (dB)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True)

    # Mostrar porcentajes en el primer gráfico
    for label, percentage in percentages_start.items():
        axes[0].annotate(f'{label}: {percentage:.2f}%', xy=(0.75, 0.9 - list(percentages_start.keys()).index(label) * 0.05), xycoords='axes fraction')

    # Histograma para rsrq_end
    axes[1].hist(df[column_end], bins=20, edgecolor='black')
    axes[1].set_title('Histograma de RSRQ End')
    axes[1].set_xlabel('RSRQ End (dB)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True)

    # Mostrar porcentajes en el segundo gráfico
    for label, percentage in percentages_end.items():
        axes[1].annotate(f'{label}: {percentage:.2f}%', xy=(0.75, 0.9 - list(percentages_end.keys()).index(label) * 0.05), xycoords='axes fraction')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()