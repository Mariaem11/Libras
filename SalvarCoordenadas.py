import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
numero = 0


# Função para calcular a diferença das coordenadas relativas
def get_relative_coordinates(hand_landmarks):
    # Ponto de referência: o pulso (landmark 0)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Lista para armazenar as coordenadas relativas
    relative_coords = []
    letra = 1
    relative_coords.append(f"{letra:.0f}")
    # Iterar sobre todos os 21 pontos de mão
    for landmark in hand_landmarks.landmark:
        # Coordenadas absolutas
        x_abs = landmark.x
        y_abs = landmark.y
        z_abs = landmark.z if hasattr(landmark, 'z') else 0

        # Coordenadas relativas em relação ao pulso
        x_rel = x_abs - wrist.x
        y_rel = y_abs - wrist.y
        z_rel = z_abs - wrist.z if hasattr(landmark, 'z') else 0  # Para 3D, calcula a diferença de z também

        # Adiciona as coordenadas relativas à lista

        relative_coords.append(f"{x_rel:.4f},{y_rel:.4f},{z_rel:.4f}")

    return relative_coords


# Abrir o arquivo de texto para salvar as coordenadas
with open("C:\\Users\\User\\PycharmProjects\\Libras\\.venv\\Letras\\Letra_A.txt", "a") as file:
    # Inicialização do MediaPipe Hands
    webcam = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while webcam.isOpened():
            success, image = webcam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Converte a imagem para o formato RGB
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Deixe a imagem gravável novamente
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Verifica se existem mãos detectadas
            if results.multi_hand_landmarks:
                # Apenas pega a primeira mão detectada
                hand_landmarks = results.multi_hand_landmarks[0]  # Processa apenas a primeira mão

                # Obtém as coordenadas relativas
                relative_coords = get_relative_coordinates(hand_landmarks)

                # Quando a tecla "j" for pressionada, salva as coordenadas no arquivo
                key = cv2.waitKey(1)
                if key == ord("j"):  # Se pressionar 'j', salva as coordenadas
                    # Escreve as coordenadas relativas em uma linha no arquivo de texto
                    file.write(",".join(relative_coords) + "\n")
                    print("Coordenadas relativas salvas:", relative_coords)

                # Desenha os landmarks na imagem
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Exibe a imagem processada
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            key = cv2.waitKey(5)
            # Sai do loop se a tecla ESC for pressionada
            if key == 27 or cv2.getWindowProperty("MediaPipe Hands", cv2.WND_PROP_VISIBLE) < 1:
                break

    # Libera a câmera e fecha a janela
    webcam.release()
    cv2.destroyAllWindows()
