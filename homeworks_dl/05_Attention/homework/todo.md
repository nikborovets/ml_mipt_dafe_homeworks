# TODO для HW03

## Задание 1: Генератор суммаризации
- [x] 1.1 Написать функцию `generate_summary(model, field, src_text)`  
- [ ] 1.2 Вывести 5 примеров из тестового датасета с ROUGE  
- [ ] 1.3 Написать 5 собственных примеров и вывести для них суммаризацию

## Задание 2: ROUGE-метрика
- [x] 2.1 Подключить `rouge_scorer.RougeScorer`  
- [x] 2.2 Интегрировать расчёт ROUGE в `train.py` (во время обучения)  
- [x] 2.3 Интегрировать расчёт ROUGE в `evaluate.py` (после тренировки)

## Задание 3: Визуализация внимания
- [x] 3.1 Извлечь матрицы attention (`attn_probs`) из модели  
- [x] 3.2 Написать функцию `plot_attention(attn_matrix, src_tokens, tgt_tokens)`  
- [ ] 3.3 Проверить визуализацию на 3 примерах и сохранить графики в `docs/attention_examples/`

## Задание 4: Общие матрицы эмбеддингов
- [x] 4.1 Вынести создание эмбеддингов в класс `SharedEmbeddings`  
- [x] 4.2 Заменить в `Encoder` и `Decoder` прямые `nn.Embedding` на `SharedEmbeddings`  
- [x] 4.3 Проверить, что выходной слой декодера тоже использует ту же матрицу

## Задание 5: Label Smoothing
- [x] 5.1 Написать класс `LabelSmoothingLoss` в `src/train.py`  
- [x] 5.2 Заменить стандартный `nn.NLLLoss` на `LabelSmoothingLoss` в тренировочном цикле  
- [x] 5.3 Написать тест `test_loss.py` для проверки корректности Label Smoothing

## Задание 6: Предобученные эмбеддинги для русского
- [ ] 6.1 Найти и скачать эмбеддинги русского (FastText или RUSSE-BERT) через MCP-сервер context7  
- [ ] 6.2 Загрузить эмбеддинги в `SharedEmbeddings`  
- [ ] 6.3 Сравнить метрики (ROUGE) «до» и «после» и записать результаты в `docs/embedding_comparison.md`

## Дополнительные задачи (выполнены)
- [x] Исправить ошибки в тестах
- [x] Исправить размерности масок в MultiHeadedAttention
- [x] Исправить тест Label Smoothing Loss
- [x] Исправить тест Noam optimizer
- [x] Унифицировать выбор device с приоритетом mps -> cuda -> cpu

## Финальная упаковка
- [ ] Сохранить `best_model.pt`  
- [ ] Сохранить `training_plot.png`  
- [ ] Создать `predictions_on_test.txt` для 10 примеров  
- [ ] Собрать архив `BorovetsNV-hw03.zip`