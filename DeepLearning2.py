import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para carregar e pré-processar os dados
def load_data_from_txt(data_file):
    # Ler o arquivo .txt, garantindo que o delimitador está correto
    try:
        data = pd.read_csv(data_file, header=None, delimiter=',', on_bad_lines='skip')
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None, None

    # O primeiro valor é o rótulo (label), os outros são as coordenadas (X, Y, Z)
    X = data.iloc[:, 1:].values  # As coordenadas dos pontos (60 valores)
    y = data.iloc[:, 0].values  # O rótulo (primeira coluna)

    # Normalização das coordenadas (opcional, mas recomendado)
    X = X / np.max(np.abs(X), axis=0)  # Normaliza as coordenadas para [-1, 1]

    return X, y


# Caminho para o arquivo de dados
data_file = 'C:\\Users\\User\\PycharmProjects\\Libras\\dataset\\Letras_modifi.txt'  # Ajuste para o caminho correto

# Carregar os dados
X, y = load_data_from_txt(data_file)

if X is not None and y is not None:
    print(f"Dados carregados com sucesso! Tamanho de X: {X.shape}, Tamanho de y: {y.shape}")
else:
    print("Falha ao carregar os dados.")

# Caminho para o arquivo de dados
data_file = 'C:\\Users\\User\\Desktop\\Letras_modifi.txt'  # Ajuste para o caminho correto

# Carregar os dados
X, y = load_data_from_txt(data_file)

# Codificação das etiquetas (one-hot encoding) se os rótulos forem categorias
lb = LabelBinarizer()
y = lb.fit_transform(y)  # Converte os rótulos para vetor one-hot

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar o modelo da rede neural
model = Sequential([
    Dense(128, activation='relu', input_dim=X.shape[1]),  # 60 valores de entrada (coordenadas X, Y, Z)
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')  # O número de saídas é o número de classes (rótulos)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Salvar o modelo treinado
model.save('C:\\Users\\User\\Desktop\\letra_rec_mlp2.keras')
