import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from itertools import islice

# Важно: Убедитесь, что этот импорт работает и src/face_verifier.py на месте
from src.face_verifier import FaceVerifier

# --- КОНСТАНТЫ ---
LFW_HOME_PATH = './data/lfw_home'
THRESHOLD_SAVE_PATH = './models/verification_threshold.json'
GRAPH_SAVE_PATH = './reports/accuracy_vs_threshold.png'
# Количество пар для обработки. Можно уменьшить до 200-300 для быстрой отладки.
NUM_PAIRS_TO_PROCESS = 1000


# ==============================================================================
# === НАШ СОБСТВЕННЫЙ ЗАГРУЗЧИК ДАННЫХ LFW - ОБХОДИМ ПРОБЛЕМУ С TORCHVISION ===
# ==============================================================================
def load_lfw_pairs_manually(root_path, pairs_file_name="pairs.txt"):
    """
    Читает файл pairs.txt и генерирует пары путей к изображениям и их метки.
    Это замена неработающему torchvision.datasets.LFWPairs.
    """
    pairs_path = os.path.join(root_path, pairs_file_name)
    lfw_images_path = os.path.join(root_path, 'lfw')

    with open(pairs_path, 'r') as f:
        # Пропускаем первую строку с метаданными
        next(f)
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) == 3:  # Пара одного и того же человека
                name, num1, num2 = parts
                path1 = os.path.join(lfw_images_path, name, f"{name}_{int(num1):04d}.jpg")
                path2 = os.path.join(lfw_images_path, name, f"{name}_{int(num2):04d}.jpg")
                label = 1  # 1 - это один и тот же человек
            elif len(parts) == 4:  # Пара разных людей
                name1, num1, name2, num2 = parts
                path1 = os.path.join(lfw_images_path, name1, f"{name1}_{int(num1):04d}.jpg")
                path2 = os.path.join(lfw_images_path, name2, f"{name2}_{int(num2):04d}.jpg")
                label = 0  # 0 - это разные люди
            else:
                continue

            if os.path.exists(path1) and os.path.exists(path2):
                yield path1, path2, label


# ==============================================================================

def find_best_threshold():
    """
    Основная функция для вычисления порога, использующая наш собственный загрузчик.
    """
    os.makedirs(os.path.dirname(THRESHOLD_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(GRAPH_SAVE_PATH), exist_ok=True)

    verifier = FaceVerifier()

    print(f"Загрузка пар изображений вручную из: {os.path.abspath(LFW_HOME_PATH)}")
    pair_generator = load_lfw_pairs_manually(LFW_HOME_PATH)

    pairs_to_process = list(islice(pair_generator, NUM_PAIRS_TO_PROCESS))
    print(f"Будет обработано {len(pairs_to_process)} пар.")

    print("Вычисление расстояний...")
    distances = []
    labels = []

    for path1, path2, label in tqdm(pairs_to_process):
        emb1 = verifier.get_embedding(path1)
        emb2 = verifier.get_embedding(path2)

        dist = verifier.calculate_distance(emb1, emb2)
        distances.append(dist)
        labels.append(label)

    print("Поиск оптимального порога...")
    thresholds = np.arange(0.4, 2.0, 0.01)
    accuracies = []

    for t in thresholds:
        predictions = [1 if d < t else 0 for d in distances]
        acc = accuracy_score(labels, predictions)
        accuracies.append(acc)

    best_accuracy = max(accuracies)
    best_threshold = thresholds[np.argmax(accuracies)]
    print(f"Лучший порог: {best_threshold:.4f} с точностью {best_accuracy:.4f}")

    with open(THRESHOLD_SAVE_PATH, 'w') as f:
        json.dump({'best_threshold': best_threshold}, f)
    print(f"Порог сохранен в {THRESHOLD_SAVE_PATH}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies)
    plt.title('Зависимость точности от порога на датасете LFW')
    plt.xlabel('Порог расстояния')
    plt.ylabel('Точность (Accuracy)')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Лучший порог = {best_threshold:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"График сохранен в {GRAPH_SAVE_PATH}")


if __name__ == "__main__":
    find_best_threshold()