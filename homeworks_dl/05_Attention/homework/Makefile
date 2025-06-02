.PHONY: install test train evaluate clean package

# Установка зависимостей
install:
	source .venv/bin/activate && pip install -r requirements.txt

# Запуск тестов
test:
	source .venv/bin/activate && python -m pytest tests/ -v

# Обработка данных
preprocess:
	source .venv/bin/activate && python src/data.py

# Обучение модели
train:
	source .venv/bin/activate && python src/train.py

# Оценка модели
evaluate:
	source .venv/bin/activate && python src/evaluate.py

# Полный пайплайн
pipeline: preprocess train evaluate

# Очистка временных файлов
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -f *.pyc src/*.pyc tests/*.pyc

# Создание архива для сдачи
package:
	bash package.sh

# Запуск DVC пайплайна
dvc-repro:
	source .venv/bin/activate && dvc repro

# Инициализация DVC
dvc-init:
	source .venv/bin/activate && dvc init 