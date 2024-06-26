import pandas as pd
import ast
import numpy as np

band_freq_from_band_dict = {
    '5G': {
        1: 'NR-SDK-2100',
        2: 'NR-SDK-1900',
        3: 'NR-SDK-1800',
        5: 'NR-SDK-850',
        7: 'NR-SDK-2600',
        8: 'NR-SDK-900',
        12: 'NR-SDK-700',
        13: 'NR-SDK-700',
        24: 'NR-SDK-1500',
        25: 'NR-SDK-1900',
        28: 'NR-SDK-700',
        29: 'NR-SDK-700',
        34: 'NR-SDK-2000',
        38: 'NR-SDK-2600',
        40: 'NR-SDK-2300',
        41: 'NR-SDK-2500',
        50: 'NR-SDK-1500',
        53: 'NR-SDK-2500',
        66: 'NR-SDK-2100',
        70: 'NR-SDK-2000',
        71: 'NR-SDK-600',
        77: 'NR-SDK-3700',
        78: 'NR-SDK-3500',
        79: 'NR-SDK-4700',
        96: 'NR-SDK-6000',
        100: 'NR-SDK-900',
        101: 'NR-SDK-1900',
        258: 'NR-SDK-26000',
        259: 'NR-SDK-41000',
        260: 'NR-SDK-39000',
        261: 'NR-SDK-28000',
        262: 'NR-SDK-48000'
    },
    '4G': {
        1: 'LTE-SDK-2100',
        2: 'LTE-SDK-1900',
        3: 'LTE-SDK-1800',
        4: 'LTE-SDK-2100',
        5: 'LTE-SDK-850',
        7: 'LTE-SDK-2600',
        8: 'LTE-SDK-900',
        9: 'LTE-SDK-1800',
        10: 'LTE-SDK-2100',
        11: 'LTE-SDK-1500',
        12: 'LTE-SDK-700',
        13: 'LTE-SDK-700',
        14: 'LTE-SDK-700',
        17: 'LTE-SDK-700',
        18: 'LTE-SDK-800',
        19: 'LTE-SDK-800',
        20: 'LTE-SDK-800',
        21: 'LTE-SDK-1500',
        22: 'LTE-SDK-3500',
        23: 'LTE-SDK-2000',
        24: 'LTE-SDK-1600',
        25: 'LTE-SDK-1900',
        26: 'LTE-SDK-850',
        27: 'LTE-SDK-800',
        28: 'LTE-SDK-700',
        29: 'LTE-SDK-700',
        30: 'LTE-SDK-2300',
        31: 'LTE-SDK-450',
        32: 'LTE-SDK-1500',
        33: 'LTE-SDK-1900',
        34: 'LTE-SDK-2000',
        35: 'LTE-SDK-1800',
        36: 'LTE-SDK-1900',
        37: 'LTE-SDK-1900',
        38: 'LTE-SDK-2600',
        39: 'LTE-SDK-1900',
        40: 'LTE-SDK-2300',
        41: 'LTE-SDK-2600',
        42: 'LTE-SDK-3500',
        43: 'LTE-SDK-3700',
        44: 'LTE-SDK-700',
        45: 'LTE-SDK-1500',
        46: 'LTE-SDK-5500',
        47: 'LTE-SDK-5900',
        48: 'LTE-SDK-3600',
        49: 'LTE-SDK-3600',
        50: 'LTE-SDK-1500',
        51: 'LTE-SDK-1500',
        52: 'LTE-SDK-3300',
        53: 'LTE-SDK-2500',
        65: 'LTE-SDK-2100',
        66: 'LTE-SDK-2100',
        67: 'LTE-SDK-700',
        68: 'LTE-SDK-700',
        69: 'LTE-SDK-2600',
        70: 'LTE-SDK-2000',
        71: 'LTE-SDK-600',
        72: 'LTE-SDK-450',
        73: 'LTE-SDK-450',
        74: 'LTE-SDK-1500',
        75: 'LTE-SDK-1500',
        76: 'LTE-SDK-1500',
        85: 'LTE-SDK-700',
        87: 'LTE-SDK-410',
        88: 'LTE-SDK-410',
        252: 'LTE-SDK-5200',
        255: 'LTE-SDK-5800'
    },
    '3G': {
        1: 'UMTS-SDK-2100',
        2: 'UMTS-SDK-1900',
        3: 'UMTS-SDK-1800',
        4: 'UMTS-SDK-2100',
        5: 'UMTS-SDK-850',
        6: 'UMTS-SDK-850',
        7: 'UMTS-SDK-2600',
        8: 'UMTS-SDK-900',
        9: 'UMTS-SDK-1800',
        10: 'UMTS-SDK-2100',
        11: 'UMTS-SDK-1500',
        12: 'UMTS-SDK-700',
        13: 'UMTS-SDK-700',
        14: 'UMTS-SDK-700',
        19: 'UMTS-SDK-800',
        20: 'UMTS-SDK-800',
        21: 'UMTS-SDK-1500',
        22: 'UMTS-SDK-3500',
        25: 'UMTS-SDK-1900',
        26: 'UMTS-SDK-850',
        32: 'UMTS-SDK-1500',
        33: 'UMTS-SDK-1900',
        34: 'UMTS-SDK-2000',
        35: 'UMTS-SDK-1900',
        36: 'UMTS-SDK-1900',
        37: 'UMTS-SDK-1900',
        38: 'UMTS-SDK-2600',
        39: 'UMTS-SDK-1900',
        40: 'UMTS-SDK-2300'
    },
    '2G': {
        (259, 293): 'GSM-SDK-450',
        (306, 340): 'GSM-SDK-480',
        (438, 511): 'GSM-SDK-750',
        (128, 251): 'GSM-SDK-850',
        (0, 124): 'GSM-SDK-900',
        (955, 1023): 'GSM-SDK-900',
        (512, 885): 'GSM-SDK-1800/1900'
    }
}

band_5G_4G_3G_dict = {
    '5G': {
        (float('-inf'), 123400): -1,
        (123400, 130400): 71,
        (143400, 145600): 29,
        (145800, 149200): 12,
        (149200, 151200): 13,
        (151600, 160600): 28,
        (173000, 178800): 5,
        (183880, 185000): 100,
        (185000, 192000): 8,
        (286400, 303400): 50,
        (305000, 311800): 24,
        (361000, 376000): 3,
        (380000, 382000): 101,
        (386000, 398000): 2,
        (398000, 399000): 25,
        (399000, 404000): 70,
        (402000, 405000): 34,
        (422000, 440000, ('us', 'ca', 'mx')): 66,
        (422000, 434000): 1,
        (460000, 480000): 40,
        (496700, 499000): 53,
        (499000, 537999): 41,
        (514000, 524000): 38,
        (524000, 538000): 7,
        (620000, 653333): 78,
        (653334, 680000): 77,
        (693334, 733332): 79,
        (795000, 875000): 96,
        (2016667, 2070832): 258,
        (2070833, 2084999): 261,
        (2229166, 2279165): 260,
        (2270832, 2337499): 259,
        (2399166, 2415832): 262
    },
    '4G': {
        (float('-inf'), 0): -1,
        (0, 599): 1,
        (600, 1199): 2,
        (1200, 1949): 3,
        (1950, 2399): 4,
        (2400, 2649): 5,
        (2650, 2749): 6,
        (2750, 3449): 7,
        (3450, 3799): 8,
        (3800, 4149): 9,
        (4150, 4749): 10,
        (4750, 4949): 11,
        (5010, 5179): 12,
        (5180, 5279): 13,
        (5280, 5379): 14,
        (5730, 5849): 17,
        (5850, 5999): 18,
        (6000, 6149): 19,
        (6150, 6449): 20,
        (6450, 6599): 21,
        (6600, 7399): 22,
        (7500, 7699): 23,
        (7700, 8039): 24,
        (8040, 8689): 25,
        (8690, 9039): 26,
        (9040, 9209): 27,
        (9210, 9659): 28,
        (9660, 9769): 29,
        (9770, 9869): 30,
        (9870, 9919): 31,
        (9920, 10359): 32,
        (36000, 36199): 33,
        (36200, 36349): 34,
        (36350, 36949): 35,
        (36950, 37549): 36,
        (37500, 37749): 37,
        (37750, 38249): 38,
        (38250, 38649): 39,
        (38650, 39649): 40,
        (39650, 41589): 41,
        (41590, 43589): 42,
        (43590, 45589): 43,
        (45590, 46589): 44,
        (46590, 46789): 45,
        (46790, 54539): 46,
        (54540, 55239): 47,
        (55240, 56739): 48,
        (56740, 58239): 49,
        (58240, 59089): 50,
        (59090, 59139): 51,
        (59140, 60139): 52,
        (60140, 60254): 53,
        (65536, 66435): 65,
        (66436, 67335): 66,
        (67336, 67535): 67,
        (67536, 67835): 68,
        (67836, 68335): 69,
        (68336, 68585): 70,
        (68586, 68935): 71,
        (68936, 68985): 72,
        (68986, 69035): 73,
        (69036, 69465): 74,
        (69466, 70315): 75,
        (70316, 70365): 76,
        (70366, 70545): 85,
        (70546, 70595): 87,
        (70596, 70645): 88,
        (255144, 256143): 252,
        (260894, 262143): 255
    },
    '3G': {
        (float('-inf'), 0): -1,
        (10562, 10838): 1,
        (9662, 9938): 2,
        (1162, 1513): 3,
        (1537, 1738): 4,
        (4357, 4458): 5,
        (2237, 2563): 7,
        (2937, 3088): 8,
        (9237, 9387): 9,
        (3112, 3388): 10,
        (3712, 3787): 11,
        (3842, 3903): 12,
        (4017, 4043): 13,
        (4117, 4143): 14,
        (712, 763): 19,
        (4512, 4638): 20,
        (862, 912): 21,
        (4662, 5038): 22,
        (5112, 5413): 25,
        (5762, 5913): 26,
        (6617, 6813): 32,
        (9500, 9600): 33,
        (10050, 10125): 34,
        (9250, 9399): 35,
        (9650, 9950): 36,
        (9550, 9650): 37,
        (12850, 13100): 38,
        (9400, 9600): 39,
        (11500, 12000): 40
    }
}

def processing_function(df_SA_4G, df_NSA):
    # Añadir columnas 'id'
    df_SA_4G['id'] = range(1, len(df_SA_4G) + 1)
    df_NSA['id'] = range(1, len(df_NSA) + 1)
    
    # Función para convertir Bytes a Mbits
    def bytes_to_mbits(x):
        try:
            if isinstance(x, str):  
                x = ast.literal_eval(x)
            return [int(i) * 8 / 1e6 for i in x]
        except Exception as e:
            print(f"Error processing {x}: {e}")
            return x 

    # Funciones aplicadas a df_SA_4G
    def assign_frequency_band(row):
        cell_network_type = row['type']
        bands = row['arfcn_start']
    
        if pd.isna(cell_network_type):
            return None
    
        network_type_str = f"{cell_network_type}G"

        for network_type, bands_dict in band_5G_4G_3G_dict.items():
            if network_type_str == network_type:
                for band_range, value in bands_dict.items():
                    if isinstance(band_range, tuple):
                        if len(band_range) == 2 and band_range[0] <= bands < band_range[1]:
                            return value
                        elif len(band_range) == 3 and band_range[0] <= bands < band_range[1] and cell_network_type in band_range[2]:
                            return value
                    else:
                        if band_range[0] <= bands < band_range[1]:
                            return value
        return None
    
    def assign_frequency_band_final(row):
        cell_network_type = row['type']
        band = row['frequency_band_start_']  # Asumimos que esto es un número
    
        network_type_str = f"{cell_network_type}G"

        # Intentamos convertir el valor a un entero, manejando excepciones para valores no convertibles
        try:
            band = int(band)
        except ValueError:
            return None  # Si no se puede convertir a int, retornamos None

        # Comprobamos si la banda es válida y devolvemos la frecuencia correspondiente
        if band in band_freq_from_band_dict.get(network_type_str, {}):
            return band_freq_from_band_dict[network_type_str][band]

        return None  # Si no encontramos ninguna banda válida, retornamos None
    
    def assign_frequency_band_end(row):
        cell_network_type = row['type']
        bands = row['arfcn_end']
    
        if pd.isna(cell_network_type):
            return None
    
        network_type_str = f"{cell_network_type}G"

        for network_type, bands_dict in band_5G_4G_3G_dict.items():
            if network_type_str == network_type:
                for band_range, value in bands_dict.items():
                    if isinstance(band_range, tuple):
                        if len(band_range) == 2 and band_range[0] <= bands < band_range[1]:
                            return value
                        elif len(band_range) == 3 and band_range[0] <= bands < band_range[1] and cell_network_type in band_range[2]:
                            return value
                    else:
                        if band_range[0] <= bands < band_range[1]:
                            return value
        return None

    def assign_frequency_band_end_final(row):
        cell_network_type = row['type']
        band = row['frequency_band_end_']  # Asumimos que esto es un número
    
        network_type_str = f"{cell_network_type}G"

        # Intentamos convertir el valor a un entero, manejando excepciones para valores no convertibles
        try:
            band = int(band)
        except ValueError:
            return None  # Si no se puede convertir a int, retornamos None

        # Comprobamos si la banda es válida y devolvemos la frecuencia correspondiente
        if band in band_freq_from_band_dict.get(network_type_str, {}):
            return band_freq_from_band_dict[network_type_str][band]

        return None  # Si no encontramos ninguna banda válida, retornamos None
    
    # Funciones aplicadas a df_NSA
    def assign_secondary_band_start(row):
        cell_network_type = row['data_coverage']
        bands = row['secondary_arfcn_start']
    
        if pd.isna(cell_network_type):
            return None
    
        network_type_str = f"{cell_network_type}G"

        for network_type, bands_dict in band_5G_4G_3G_dict.items():
            if network_type_str == network_type:
                for band_range, value in bands_dict.items():
                    if isinstance(band_range, tuple):
                        if len(band_range) == 2 and band_range[0] <= bands < band_range[1]:
                            return value
                        elif len(band_range) == 3 and band_range[0] <= bands < band_range[1] and cell_network_type in band_range[2]:
                            return value
                    else:
                        if band_range[0] <= bands < band_range[1]:
                            return value
        return None

    def assign_secondary_frequency_band_start(row):
        cell_network_type = row['data_coverage']
        band = row['secondary_band_start']  # Asumimos que esto es un número
    
        network_type_str = f"{cell_network_type}G"

        # Intentamos convertir el valor a un entero, manejando excepciones para valores no convertibles
        try:
            band = int(band)
        except ValueError:
            return None  # Si no se puede convertir a int, retornamos None

        # Comprobamos si la banda es válida y devolvemos la frecuencia correspondiente
        if band in band_freq_from_band_dict.get(network_type_str, {}):
            return band_freq_from_band_dict[network_type_str][band]

        return None  # Si no encontramos ninguna banda válida, retornamos None
    
    def assign_secondary_band_end(row):
        cell_network_type = row['data_coverage']
        bands = row['secondary_arfcn_end']
    
        if pd.isna(cell_network_type):
            return None
    
        network_type_str = f"{cell_network_type}G"

        for network_type, bands_dict in band_5G_4G_3G_dict.items():
            if network_type_str == network_type:
                for band_range, value in bands_dict.items():
                    if isinstance(band_range, tuple):
                        if len(band_range) == 2 and band_range[0] <= bands < band_range[1]:
                            return value
                        elif len(band_range) == 3 and band_range[0] <= bands < band_range[1] and cell_network_type in band_range[2]:
                            return value
                    else:
                        if band_range[0] <= bands < band_range[1]:
                            return value
        return None

    def assign_secondary_frequency_band_end(row):
        cell_network_type = row['data_coverage']
        band = row['secondary_band_end']  # Asumimos que esto es un número
    
        network_type_str = f"{cell_network_type}G"

        # Intentamos convertir el valor a un entero, manejando excepciones para valores no convertibles
        try:
            band = int(band)
        except ValueError:
            return None  # Si no se puede convertir a int, retornamos None

        # Comprobamos si la banda es válida y devolvemos la frecuencia correspondiente
        if band in band_freq_from_band_dict.get(network_type_str, {}):
            return band_freq_from_band_dict[network_type_str][band]

        return None  # Si no encontramos ninguna banda válida, retornamos None

    # Aplicar transformaciones para df_SA_4G
    df_SA_4G['dl_th_list'] = df_SA_4G['dl_th_list'].apply(bytes_to_mbits)
    df_SA_4G['frequency_band_start_'] = df_SA_4G.apply(assign_frequency_band, axis=1)
    df_SA_4G['frequency_band_start'] = df_SA_4G.apply(assign_frequency_band_final, axis=1)
    df_SA_4G['frequency_band_end_'] = df_SA_4G.apply(assign_frequency_band_end, axis=1)
    df_SA_4G['frequency_band_end'] = df_SA_4G.apply(assign_frequency_band_end_final, axis=1)
    df_SA_4G['mean'] = df_SA_4G['dl_th_list'].apply(lambda x: sum(i / 0.05 for i in x) / len(x))

    # Aplicar transformaciones para df_NSA
    df_NSA['dl_th_list'] = df_NSA['dl_th_list'].apply(bytes_to_mbits)
    df_NSA['secondary_band_start'] = df_NSA.apply(assign_secondary_band_start, axis=1)
    df_NSA['secondary_frequency_band_start'] = df_NSA.apply(assign_secondary_frequency_band_start, axis=1)
    df_NSA['secondary_band_end'] = df_NSA.apply(assign_secondary_band_end, axis=1)
    df_NSA['secondary_frequency_band_end'] = df_NSA.apply(assign_secondary_frequency_band_end, axis=1)
    df_NSA['mean'] = df_NSA['dl_th_list'].apply(lambda x: sum(i / 0.05 for i in x) / len(x))
    
    # Configurar columnas de tecnología específicas para cada DataFrame
    conditions_SA_4G = [(df_SA_4G['type'] == 5), (df_SA_4G['type'] == 4)]
    choices_SA_4G = ['5G SA', '4G']
    df_SA_4G['technology'] = np.select(conditions_SA_4G, choices_SA_4G, default='')

    conditions_NSA = [(df_NSA['type'] == 4) & (df_NSA['data_coverage'] == 5)]
    choices_NSA = ['5G NSA']
    df_NSA['technology'] = np.select(conditions_NSA, choices_NSA, default='')

    return df_SA_4G, df_NSA