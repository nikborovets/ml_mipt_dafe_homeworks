# Makefile для Prototypical Networks проекта

.PHONY: install prepare-data train test coverage tensorboard clean help

# Установка зависимостей
install:
	pip install -r requirements.txt

# Подготовка данных (загрузка и обработка Omniglot)
prepare-data:
	python -m src.data --prepare
	@echo "Data preparation completed!"

# Обучение модели
train:
	python -m src.train --config config.yaml

# Тестирование модели
test:
	python -m src.evaluate --model_path models/protonet.pt --visualize

# Запуск тестов pytest
coverage:
	pytest --maxfail=1 --disable-warnings -q tests/

# Запуск TensorBoard
tensorboard:
	tensorboard --logdir runs/omniglot_protonet

# Быстрое обучение для тестирования
quick-train:
	python -m src.train --max_epoch 1 --epoch_size 100

# Быстрое тестирование
quick-test:
	python -m src.evaluate --model_path models/protonet.pt --test_episodes 10

# Загрузка данных без обработки
download-data:
	python -m src.data --prepare

# Тестирование загрузки данных
test-data:
	python -m src.data --test

# Очистка временных файлов
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/

# Очистка всех артефактов (модели, логи, метрики)
clean-all: clean
	rm -rf models/
	rm -rf runs/
	rm -rf metrics/
	rm -rf data/

# Создание архива для отправки
package:
	bash package.sh

# Проверка стиля кода
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

# Форматирование кода
format:
	black src/ tests/ --line-length=100

# Полный пайплайн: установка -> подготовка данных -> обучение -> тестирование
full-pipeline: install prepare-data train test

# Справка
help:
	@echo "Доступные команды:"
	@echo "  install        - Установить зависимости"
	@echo "  prepare-data   - Подготовить данные Omniglot"
	@echo "  train          - Обучить модель"
	@echo "  test           - Протестировать модель"
	@echo "  coverage       - Запустить pytest тесты"
	@echo "  tensorboard    - Запустить TensorBoard"
	@echo "  quick-train    - Быстрое обучение (1 эпоха)"
	@echo "  quick-test     - Быстрое тестирование (10 эпизодов)"
	@echo "  download-data  - Загрузить данные"
	@echo "  test-data      - Протестировать загрузку данных"
	@echo "  clean          - Очистить временные файлы"
	@echo "  clean-all      - Очистить все артефакты"
	@echo "  package        - Создать архив для отправки"
	@echo "  lint           - Проверить стиль кода"
	@echo "  format         - Отформатировать код"
	@echo "  full-pipeline  - Полный пайплайн от установки до тестирования"
	@echo "  help           - Показать эту справку" 