def split_data(data, split_ratio = 0.8):

	# Calcular el índice de corte (80% del total)
    split_index = int(len(data) * split_ratio)
    
    # Dividir: primeros 80% para train, últimos 20% para test
    train = data[:split_index]
    test = data[split_index:]
    
    return train, test


