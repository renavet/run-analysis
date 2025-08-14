import pandas as pd
import numpy as np

def converti_tempo_in_secondi(tempo_str):
    if pd.isna(tempo_str):
        return None
    try:
        # Se il valore è già numerico (es. 27.1), gestiscilo qui
        return float(tempo_str)
    except ValueError:
        # Se il valore è una stringa (es. "4:27"), prosegui con la conversione
        parti = str(tempo_str).split(':')
        if len(parti) == 3:
            # Formato hh:mm:ss
            h, m, s = map(float, parti)
            return h * 3600 + m * 60 + s
        elif len(parti) == 2:
            # Formato mm:ss
            m, s = map(float, parti)
            return m * 60 + s
    return None