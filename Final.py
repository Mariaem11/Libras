import cv2  # OpenCV
import mediapipe as mp  # Usado para a identificação e retorno das informações das mãos
import numpy as np
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model('C:\\Users\\User\\Desktop\\letra_rec_mlp2.keras') #Mudar o diretório

# Inicialização do MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Função para obter as coordenadas relativas das mãos (sem as coordenadas do pulso)
def get_relative_coordinates(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]  # Pulso é considerado como ponto de referência
    relative_coords = []

    for i, landmark in enumerate(hand_landmarks.landmark):
        # Exclui o pulso (não adiciona as coordenadas relativas do pulso à lista)
        if i == mp_hands.HandLandmark.WRIST:
            continue  # Pula a iteração do pulso

        # Calcula as coordenadas relativas usando o pulso como referência
        x_rel = landmark.x - wrist.x
        y_rel = landmark.y - wrist.y
        z_rel = landmark.z - wrist.z if hasattr(landmark, 'z') else 0

        # Adiciona as coordenadas relativas dos outros pontos à lista
        relative_coords.append([x_rel, y_rel, z_rel])

    return np.array(relative_coords, dtype=np.float32)

# Inicialização da webcam
webcam = cv2.VideoCapture(0)

# Inicialização do MediaPipe Hands
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Converte a imagem para RGB para o MediaPipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Verifica se há mãos detectadas
        if results.multi_hand_landmarks:
            # Processa a primeira mão detectada
            hand_landmarks = results.multi_hand_landmarks[0]

            # Obtém as coordenadas relativas das mãos
            relative_coords = get_relative_coordinates(hand_landmarks)

            # Faz a previsão com o modelo (certifique-se de que o formato da entrada esteja correto)
            relative_coords = relative_coords.flatten().reshape(1, -1)  # Ajusta o formato para o modelo
            prediction = model.predict(relative_coords)

            # O modelo retorna uma distribuição de probabilidades (softmax), obtemos a classe com maior probabilidade
            predicted_class = np.argmax(prediction, axis=1)[0]

            # A lista de labels deve ser a mesma que você usou no treinamento (A-Z, ou outras classes)
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
            #labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

            predicted_label = labels[predicted_class]

            # Exibe o label previsto na tela
            cv2.putText(image, f'Predicted: {predicted_label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Desenha os landmarks da mão na imagem
         #   '''
            mp_drawing.draw_landmarks(  #Se quiser testar sem as landmarks do media pipe é só comentar essa parte
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            '''#'''
        # Exibe a imagem com a previsão
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # Verifica se o usuário pressionou a tecla "ESC" para sair
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Libera a captura e fecha as janelas
webcam.release()
cv2.destroyAllWindows()
