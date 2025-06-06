.PHONY: install find-threshold run-demo clean

# Установить все зависимости
install:
	pip install -r requirements.txt

# Найти оптимальный порог (аналог "обучения")
find-threshold:
	python find_threshold.py

# Запустить демонстрацию на тестовых данных
run-demo:
	python run_demo.py

# Очистить сгенерированные файлы
clean:
	rm -f models/verification_threshold.json
	rm -f reports/accuracy_vs_threshold.png
	rm -f predictions.csv
	rm -rf data/lfw_home/