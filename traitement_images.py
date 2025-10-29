import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
def get_x_bounds(forme):
    coords = cv2.findNonZero(255 - forme)  # On inverse pour obtenir les pixels noirs
    x, _, w, _ = cv2.boundingRect(coords)
    return x, x + w  # x_min, x_max

def inserer_espaces(formes):

    formes_triees = sorted(formes, key=lambda f: get_x_bounds(f)[0])

    bornes = [get_x_bounds(f) for f in formes_triees]
    distances = [bornes[i+1][0] - bornes[i][1] for i in range(len(bornes) - 1)]

    moyenne_distance = np.mean(distances)
    print(f" moyenne des distances : {moyenne_distance:.2f}")

    espace_blanc = np.full((28, 28), 255, dtype=np.uint8)
    nouvelles_formes = [formes_triees[0]]

    for i in range(1, len(formes_triees)):
        dist = bornes[i][0] - bornes[i - 1][1]
        if dist > 1.3 * moyenne_distance:
            print(f" Eepace entre {i - 1} et {i} (distance = {dist})")
            nouvelles_formes.append(espace_blanc.copy())
        nouvelles_formes.append(formes_triees[i])

    return nouvelles_formes
def centrer_formes(formes):
    formes_28x28 = []
    for f in formes:
        if np.all(f == 255):  # Espace
            formes_28x28.append(np.full((28, 28), 255, dtype=np.uint8))
            continue

        # Trouver la bounding box autour des pixels noirs
        coords = cv2.findNonZero(255 - f)
        x, y, w, h = cv2.boundingRect(coords)

        # Rogner autour de la forme
        lettre_crop = f[y:y+h, x:x+w]

        # Redimensionner la lettre pour qu'elle tienne dans une boîte de 20x20
        if w > h:
            new_w = 22
            new_h = int(h * (22 / w))
        else:
            new_h = 22
            new_w = int(w * (22 / h))
        lettre_redim = cv2.resize(lettre_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Créer une nouvelle image blanche de 28x28
        canvas = np.full((28, 28), 255, dtype=np.uint8)

        # Calculer les positions pour centrer la lettre
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        # Coller la lettre redimensionnée au centre
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = lettre_redim

        formes_28x28.append(canvas)
    return formes_28x28

def get_extrema_y(image):
    """Retourne les bornes gauche et droite (x_min, x_max) de la forme noire dans l image"""
    height, width = image.shape
    y_min = width
    y_max = 0

    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:  # pixel noir (fait partie de la forme)
                if j < y_min:
                    y_min = j
                if j > y_max:
                    y_max = j

    return y_min, y_max
def dfs(x, y, binary, visited, shape_img):
    height, width = binary.shape
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()

        if not (0 <= cx < height and 0 <= cy < width):
            continue
        if visited[cx, cy]:
            continue
        if binary[cx, cy] == 0:
            continue

        visited[cx, cy] = True
        shape_img[cx, cy] = 0  # tracer la forme en noir sur fond blanc

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    stack.append((cx + dx, cy + dy))

def extraire_formes(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = 255 - img  # On inverse : fond blanc, lettres noires

    # seuillage automatique : transforme l’image en binaire (noir/blanc)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    plt.imshow(binary, cmap='gray')
    plt.title("initiale")
    plt.axis("off")
    plt.show()

    # dilatation pour combler les petits trous
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    plt.imshow(binary, cmap='gray')
    plt.title("Après dilatation")
    plt.axis("off")
    plt.show()

    # fermeture morphologique pour améliorer la netteté des lettres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    plt.imshow(binary, cmap='gray')
    plt.title("Après fermeture morphologique et dilatation")
    plt.axis("off")
    plt.show()

    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    formes = []

    # détection des formes à partir des pixels blancs non encore visités
    for y in range(width):
        for x in range(height):
            if binary[x, y] > 0 and not visited[x, y]:
                shape_img = np.full((height, width), 255, dtype=np.uint8)
                dfs(x, y, binary, visited, shape_img)
                formes.append(shape_img)

    print(f"{len(formes)} formes détectées.")

    # compte le nombre de pixels noirs par forme
    nb_pixels_par_forme = [np.sum(f == 0) for f in formes]
    moyenne = np.mean(nb_pixels_par_forme)
    seuil = moyenne / 4.0

    # on enlève les trop petites formes
    formes_filtrees = []
    for i in range(len(formes)):
        if nb_pixels_par_forme[i] >= seuil:
            formes_filtrees.append(formes[i])
        else:
            print(f" forme {i} supprimée (elle était trop petite : {nb_pixels_par_forme[i]} pixels...)")

    # on ajoute des espaces si besoin
    formes_finales = inserer_espaces(formes_filtrees)

    #on renvoie les formes recentrées
    return centrer_formes(formes_finales)


formes = extraire_formes("lignes_simple/ligne_1.png")

formes = [(255.0 - f.astype(np.float32)) / 255.0 for f in formes]

for i, f in enumerate(formes):
    plt.imshow(f)
    plt.title(f"Forme {i}")
    plt.axis("off")
    plt.show()


output_dir = "formes_sortie"
os.makedirs(output_dir)

# Sauvegarde de chaque forme en image PNG pour visualisation de l'utilisateur
for i, f in enumerate(formes):
    nom = f"espace_{i}.png" if np.all(f == 1.0) else f"lettre_{i}.png"
    Image.fromarray((f * 255).astype(np.uint8)).save(os.path.join(output_dir, nom))

# Sauvegarde des formes en texte pour récuperer dans le code C
with open("formes.txt", "w") as txt:
    nb_lettres = len(formes)
    txt.write(f"{nb_lettres}\n")  # Première ligne : nombre total de formes

    for img in formes:
        if np.all(img == 1.0):  # Si c’est un espace, on note "NULL"
            txt.write("NULL\n")
        else:
            for row in img:
                txt.write(" ".join(f"{pixel:.3f}" for pixel in row) + "\n")
            txt.write("\n")  # Ligne vide entre deux blocs
