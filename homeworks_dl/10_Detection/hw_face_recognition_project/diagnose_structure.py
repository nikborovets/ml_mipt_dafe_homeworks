import os

# --- Пути, которые мы проверяем ---
LFW_HOME_PATH = './data/lfw_home'
FINAL_IMG_PATH = os.path.join(LFW_HOME_PATH, 'lfw')
FINAL_PAIRS_PATH = os.path.join(LFW_HOME_PATH, 'pairs.txt')

print("=" * 30)
print("=== ЗАПУСК ДИАГНОСТИКИ СТРУКТУРЫ ДАТАСЕТА ===")
print("=" * 30)

# 1. Проверяем существование основной папки
print(f"\n1. Проверяем основную папку: {os.path.abspath(LFW_HOME_PATH)}")
if os.path.exists(LFW_HOME_PATH):
    print("   [OK] Папка существует.")
else:
    print("   [ОШИБКА] Папка НЕ СУЩЕСТВУЕТ. Проверьте путь.")
    exit()

# 2. Проверяем существование папки с изображениями
print(f"\n2. Проверяем папку с изображениями: {os.path.abspath(FINAL_IMG_PATH)}")
if os.path.exists(FINAL_IMG_PATH):
    print("   [OK] Папка существует.")
else:
    print("   [ОШИБКА] Папка 'lfw' внутри 'lfw_home' НЕ НАЙДЕНА. Это основная проблема.")
    exit()

# 3. Проверяем, что папка с изображениями не пустая
files_in_lfw = os.listdir(FINAL_IMG_PATH)
print(f"\n3. Проверяем содержимое папки '{FINAL_IMG_PATH}'")
if len(files_in_lfw) > 0:
    print(f"   [OK] Внутри найдено {len(files_in_lfw)} элементов (папок с именами).")
    print(f"   Пример содержимого: {files_in_lfw[:5]}")  # Показываем первые 5 для примера
else:
    print("   [ОШИБКА] Папка 'lfw' ПУСТАЯ.")
    exit()

# 4. Проверяем существование файла pairs.txt
print(f"\n4. Проверяем файл с парами: {os.path.abspath(FINAL_PAIRS_PATH)}")
if os.path.exists(FINAL_PAIRS_PATH):
    print("   [OK] Файл существует.")
else:
    print("   [ОШИБКА] Файл 'pairs.txt' НЕ НАЙДЕН.")
    exit()

# 5. Проверяем, что файл pairs.txt не пустой
file_size = os.path.getsize(FINAL_PAIRS_PATH)
print(f"\n5. Проверяем размер файла 'pairs.txt'")
if file_size > 0:
    print(f"   [OK] Размер файла {file_size} байт (не пустой).")
else:
    print("   [ОШИБКА] Файл 'pairs.txt' ПУСТОЙ (размер 0 байт).")
    exit()

print("\n" + "=" * 30)
print("=== ДИАГНОСТИКА ЗАВЕРШЕНА ===")
print("=" * 30)
print("\nЕсли все проверки [OK], но ошибка 'Dataset not found' остается,")
print("возможно, проблема в правах доступа к файлам или в скрытых файлах.")