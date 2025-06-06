# Игра в Flappy Bird (Reinforcement Learning)


## Состав команды

- Боровец Николай
- Бочкарев Богдан
- Федоров Всеволод

```bash
# запустить весь пайплайн
dvc repro
# только обучение
dvc repro train
# посмотреть граф зависимостей
dvc dag

# запуск отдельных этапов
dvc repro prepare
dvc repro test
dvc repro validate
```

```bash
# запустить обучение
python smart_dqn_agent.py --train --episodes 1300
# запустить тестирование
python smart_dqn_agent.py --test models/smart_agent_best_score_100_20250606_120000.pt
# запустить обучение с загрузкой модели
python smart_dqn_agent.py --load_and_train models/smart_agent_best_score_100_20250606_120000.pt --episodes 1300
```

```bash
# запустить pytest
python -m pytest tests/ -v
```