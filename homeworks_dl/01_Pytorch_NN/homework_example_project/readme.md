# Домашнее задание

| Дедлайн | 21.05.2025 |
| :----: | :---: |
| Номер ДЗ | ``01``   |

# Постановка задачи

Внедрите правильные тесты для обучающего конвейера. В частности, взгляните на файл test_basic.py в example_project: он содержит тесты, некоторые из которых завершаются неудачей, иногда не выполняются или явно некорректны. Ваша задача состоит в том, чтобы идентифицировать такие тесты и заставить набор тестов проходить детерминированно: это потребует изменений как в исходном коде обучения, так и в тестах, поскольку некоторые части тестового кода также должны быть изменены.

*В качестве исходного кода по обучению модели взять материалы из семинара.*

После этого реализуйте функцию test_training в test_pipeline.py , которая запускает интеграционный тест для всей процедуры обучения с разными гиперпараметрами и ожидает разные результаты. Этот тест должен увеличить охват вашего кода (измеренный с помощью pytest-cov) до >80%. 

Важно отметить, что вы должны убедиться, что ваш тестовый код, запускающий реальную модель, может выполняться как на CPU, так и на GPU.


# Запуск

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


python prepare_data.py
python train.py
python compute_metrics.py

python -m pytest tests/ --cov=. --cov-report=term-missing
python -m pytest tests/ --cov --cov-report=term-missing
```
# Результаты тестов
```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
=========================================================== test session starts ===========================================================
platform darwin -- Python 3.13.3, pytest-8.3.5, pluggy-1.6.0
rootdir: /Users/nikolayborovets/Desktop/FALT_MIPT/ML/ml_mipt_dafe_homeworks/homeworks_dl/01_Pytorch_NN/example_project
plugins: hydra-core-1.3.2, cov-6.1.1
collected 31 items                                                                                                                        

tests/test_basic.py .......                                                                                                         [ 22%]
tests/test_compute_metrics.py ..                                                                                                    [ 29%]
tests/test_hparams.py .........                                                                                                     [ 58%]
tests/test_pipeline.py ..s.......                                                                                                   [ 90%]
tests/test_prepare_data.py ...                                                                                                      [100%]

============================================================ warnings summary =============================================================
...
============================================================= tests coverage ==============================================================
____________________________________________ coverage: platform darwin, python 3.13.3-final-0 _____________________________________________

Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
compute_metrics.py                 35      6    83%   30-33, 58-60
conftest.py                         3      0   100%
hparams.py                          1      0   100%
prepare_data.py                     4      2    50%   4-5
tests/test_basic.py                24      0   100%
tests/test_compute_metrics.py      60      0   100%
tests/test_hparams.py              31      0   100%
tests/test_pipeline.py            156     20    87%   40-66, 164-167, 258
tests/test_prepare_data.py         55     24    56%   25-36, 56-74, 82-83
train.py                           86     22    74%   22-25, 48-65, 131-132, 141-164, 168
-------------------------------------------------------------
TOTAL                             455     74    84%
=============================================== 30 passed, 1 skipped, 20 warnings in 13.07s ===============================================

```