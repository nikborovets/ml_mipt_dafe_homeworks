import argparse
import json
import os
import pandas as pd

from src.face_verifier import FaceVerifier

# ==============================================================================
# ===                        ЦЕНТР УПРАВЛЕНИЯ ТЕСТОМ                         ===
# ==============================================================================
#
# Просто измените эту переменную, чтобы выбрать, кого тестировать.
# Возможные значения: "kolya", "seva", "bogdan" или "all" для всех сразу.
#
PERSON_TO_TEST = "all"  # <-- ИЗМЕНИТЕ ЭТО ЗНАЧЕНИЕ

# --- Конфигурация для каждого участника ---
TEST_SUBJECTS = {
    "kolya": {
        "ref_image": "data/test_data/kolya_reference_photo.jpg",
        "name_prefix": "kolya_"
    },
    "seva": {
        "ref_image": "data/test_data/seva_reference_photo.jpg",
        "name_prefix": "seva_"
    },
    "bogdan": {
        "ref_image": "data/test_data/bogdan_reference_photo.jpg",
        "name_prefix": "bogdan_"
    }
}


# ==============================================================================

def run_single_test(verifier, subject_name, subject_config, test_folder_path, threshold):
    """
    Запускает тест для одного конкретного человека.
    """
    print("\n" + "=" * 50)
    print(f"=== ЗАПУСК ТЕСТА ДЛЯ: {subject_name.upper()} ===")
    print("=" * 50)

    ref_image_path = subject_config["ref_image"]
    ref_embedding = verifier.get_embedding(ref_image_path)

    if ref_embedding is None:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти лицо на эталонном фото {ref_image_path}")
        return []

    print(f"Эталонное фото: {ref_image_path} ... OK")

    test_files = sorted([f for f in os.listdir(test_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    results = []
    for filename in test_files:
        test_image_path = os.path.join(test_folder_path, filename)
        test_embedding = verifier.get_embedding(test_image_path)

        distance = verifier.calculate_distance(ref_embedding, test_embedding)

        # Логика предсказания
        is_match_prediction = distance < threshold

        # Логика определения "истинной" метки для анализа
        is_true_match = filename.startswith(subject_config["name_prefix"])

        results.append({
            "subject": subject_name,
            "test_file": filename,
            "distance": f"{distance:.4f}",
            "is_true_match": "Да" if is_true_match else "Нет",
            "prediction": "Да (Совпадение)" if is_match_prediction else "Нет (Несовпадение)",
            "is_correct": is_true_match == is_match_prediction
        })

    return results


def print_summary(df):
    """
    Печатает итоговую статистику по результатам.
    """
    print("\n" + "=" * 50)
    print("=== ИТОГОВАЯ СТАТИСТИКА ===")
    print("=" * 50)

    for subject in df['subject'].unique():
        print(f"\n--- Результаты для: {subject.upper()} ---")
        subject_df = df[df['subject'] == subject]

        # 1. Проверка "Свой" (True Positives)
        my_photos_df = subject_df[subject_df['is_true_match'] == "Да"]
        correct_my_photos = my_photos_df['is_correct'].sum()
        total_my_photos = len(my_photos_df)
        print(f"  - Правильно опознаны СВОИ фото: {correct_my_photos} из {total_my_photos}")

        # 2. Проверка "Чужой" (True Negatives)
        other_photos_df = subject_df[subject_df['is_true_match'] == "Нет"]
        correct_other_photos = other_photos_df['is_correct'].sum()
        total_other_photos = len(other_photos_df)
        print(f"  - Правильно опознаны ЧУЖИЕ фото: {correct_other_photos} из {total_other_photos}")

        # Общая точность для этого человека
        total_accuracy = subject_df['is_correct'].mean()
        print(f"  - Общая точность: {total_accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Гибкая демонстрация верификации лиц.")
    parser.add_argument(
        "--test-folder",
        type=str,
        default="data/test_data/evaluation_set",
        help="Папка с тестовыми фото."
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="models/verification_threshold.json",
        help="Файл с порогом."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions_summary.csv",
        help="Имя файла для сохранения полного отчета."
    )

    args = parser.parse_args()

    # Загрузка найденного порога
    try:
        with open(args.threshold_file, 'r') as f:
            data = json.load(f)
            threshold = data['best_threshold']
    except FileNotFoundError:
        print(f"Файл с порогом '{args.threshold_file}' не найден! Запустите 'find_threshold.py' сначала.")
        exit()

    # Инициализация верификатора
    verifier = FaceVerifier()

    all_results = []

    if PERSON_TO_TEST == "all":
        # Запускаем тесты для всех
        for name, config in TEST_SUBJECTS.items():
            results = run_single_test(verifier, name, config, args.test_folder, threshold)
            all_results.extend(results)
    elif PERSON_TO_TEST in TEST_SUBJECTS:
        # Запускаем тест для одного выбранного человека
        name = PERSON_TO_TEST
        config = TEST_SUBJECTS[name]
        results = run_single_test(verifier, name, config, args.test_folder, threshold)
        all_results.extend(results)
    else:
        print(f"Ошибка: Неизвестное имя '{PERSON_TO_TEST}'. Проверьте переменную PERSON_TO_TEST.")
        exit()

    if all_results:
        # Создаем и сохраняем DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_file, index=False)
        print(f"\nПолный отчет сохранен в файл: {args.output_file}")

        # Печатаем красивую сводку в консоль
        print_summary(df)