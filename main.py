import cv2
import numpy as np

# Carga la imagen
imagen = cv2.imread('/mnt/data/rectas.jpg')

# Convierte la imagen a escala de grises para mejor detección
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplica un filtro de desenfoque para suavizar la imagen y bajar el ruido
gris_suavizado = cv2.GaussianBlur(gris, (5, 5), 0)

# Aplica un umbral para convertir la imagen en binaria en blanco y negro
_, imagen_binaria = cv2.threshold(gris_suavizado, 128, 255, cv2.THRESH_BINARY)

# Usa la transformada de Hough para identificar las líneas
lineas = cv2.HoughLines(imagen_binaria, rho=1, theta=np.pi/180, threshold=100)

# Traza las líneas identificadas en la imagen original
if lineas is not None:
    for linea in lineas:
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Devuelve la imagen con las líneas detectadas
cv2.imshow('Detección de Líneas con Transformada de Hough', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
