import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import game.wrapped_flappy_bird as game # Убедитесь, что game.wrapped_flappy_bird доступен
import matplotlib.pyplot as plt
import os
import time
import argparse
import datetime # Добавим импорт datetime для временной метки

class SmartDQN(nn.Module):
    """Улучшенная версия DQN с dropout и batch normalization"""
    
    def __init__(self, input_size=5, hidden_size=256, output_size=2):
        super(SmartDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # BatchNorm1d ожидает (N, C) или (N, C, L), где N - batch_size, C - channels (фичи)
        # Если x одномерный (только фичи), и батч размером 1, нужно добавить размерность канала
        if x.ndim == 1:
            x = x.unsqueeze(0) # (features) -> (1, features) - для случая инференса одного состояния
        
        # Если батч уже есть, но нет размерности канала (N, features)
        # BatchNorm1d ожидает (N, num_features)
        # Linear ожидает (N, *, in_features)

        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class SmartAgent:
    """Умный агент с curriculum learning и shaped rewards"""
    
    def __init__(self, state_size=5, action_size=2, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.epsilon = 0.9  # Начальное значение epsilon
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999 # Замедляем затухание epsilon еще больше
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = SmartDQN(state_size, 256, action_size).to(self.device)
        self.target_network = SmartDQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-4) # Добавлен weight_decay для регуляризации
        
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_enhanced_state(self, env):
        bird_y = env.playery / float(game.SCREENHEIGHT) # Нормализация
        bird_vel = env.playerVelY / 15.0 # Нормализация (примерная)
        
        next_pipe_dist_norm = 1.0
        next_pipe_gap_center_norm = 0.5 
        next_pipe_gap_top_norm = 0.5 # Центр экрана по умолчанию, если труб нет

        bird_center_x_abs = env.playerx + game.PLAYER_WIDTH / 2.0
        
        closest_pipe = None
        min_dist = float('inf')

        for upper_pipe in env.upperPipes:
            pipe_right_edge = upper_pipe['x'] + game.PIPE_WIDTH
            if pipe_right_edge > bird_center_x_abs: # Труба еще не полностью пройдена центром птицы
                dist_to_pipe_front = upper_pipe['x'] - bird_center_x_abs
                if dist_to_pipe_front < min_dist : # Ищем ближайшую трубу Спереди
                    # Условие dist_to_pipe_front > -game.PIPE_WIDTH/2 примерно означает, что мы еще не глубоко внутри трубы
                    if dist_to_pipe_front > -game.PIPE_WIDTH : # Чтобы не брал ту трубу, которую только что прошли
                        min_dist = dist_to_pipe_front
                        closest_pipe = upper_pipe


        if closest_pipe:
            # Находим соответствующую нижнюю трубу (они хранятся парами в self.upperPipes и self.lowerPipes)
            # Это предполагает, что трубы всегда добавляются и удаляются парами.
            try:
                pipe_index = -1
                for i, p in enumerate(env.upperPipes):
                    if p['x'] == closest_pipe['x'] and p['y'] == closest_pipe['y']:
                        pipe_index = i
                        break
                
                if pipe_index != -1:
                    lower_pipe_for_closest_upper = env.lowerPipes[pipe_index]

                    # Нормализуем расстояние до начала трубы (0, если прямо перед ней, 1 если далеко)
                    # SCREENWIDTH - это примерный максимальный видимый диапазон для труб
                    next_pipe_dist_norm = max(0, min_dist) / float(game.SCREENWIDTH) 
                    
                    gap_top_y = closest_pipe['y'] + game.PIPE_HEIGHT # y верхнего края нижней части трубы (реальный верх зазора)
                    gap_bottom_y = lower_pipe_for_closest_upper['y'] # y верхнего края нижней трубы (реальный низ зазора)
                    
                    gap_center_y = (gap_top_y + gap_bottom_y) / 2.0
                    
                    next_pipe_gap_center_norm = gap_center_y / float(game.SCREENHEIGHT)
                    next_pipe_gap_top_norm = gap_top_y / float(game.SCREENHEIGHT) # Позиция верхнего края зазора
                
            except IndexError: # На случай если что-то пошло не так с индексами (маловероятно при правильной логике игры)
                 pass


        return np.array([bird_y, bird_vel, next_pipe_dist_norm, next_pipe_gap_center_norm, next_pipe_gap_top_norm])

    def get_shaped_reward(self, env, terminal, prev_score, state_info):
        # state_info это bird_y, bird_vel, next_pipe_dist_norm, next_pipe_gap_center_norm
        bird_y_norm, _, next_pipe_dist_norm, next_pipe_gap_center_norm, next_pipe_gap_top_norm = state_info
        
        reward = 0
        
        # Базовая награда за выживание (небольшая, чтобы не поощрять простое падение)
        # reward += 0.01 # Убрал, т.к. есть в env.frame_step

        if terminal:
            return -100  # Большое наказание за смерть

        # Награда за каждый выживший кадр (помогает агенту научиться дольше оставаться в живых)
        reward += 1.0 

        if env.score > prev_score:
            return 200  # Очень большая награда за очки (прохождение трубы)

        # Награда за приближение к центру зазора следующей трубы
        # Чем ближе птица к центру зазора И чем ближе сама труба, тем выше награда
        # next_pipe_dist_norm: 0 (близко) to 1 (далеко)
        # bird_y_norm: положение птицы
        # next_pipe_gap_center_norm: центр зазора
        
        # Расстояние до центра зазора по вертикали
        vertical_dist_to_gap_center = abs(bird_y_norm - next_pipe_gap_center_norm)

        if next_pipe_dist_norm < 0.3: # Если труба относительно близко
            # Награда за нахождение в безопасной зоне зазора
            # next_pipe_gap_top_norm - это верхний край зазора (низ верхней трубы)
            # next_pipe_gap_bottom_norm можно рассчитать как next_pipe_gap_top_norm + (PIPEGAPSIZE / SCREENHEIGHT)
            pipe_gap_size_norm = game.PIPEGAPSIZE / float(game.SCREENHEIGHT)
            gap_bottom_actual_norm = next_pipe_gap_top_norm + pipe_gap_size_norm

            # Если птица между верхом и низом зазора
            if next_pipe_gap_top_norm < bird_y_norm < gap_bottom_actual_norm:
                # Чем ближе к центру, тем лучше. max reward = 10
                reward += (1.0 - (vertical_dist_to_gap_center / (pipe_gap_size_norm / 2.0))) * 10 
            else:
                # Штраф за нахождение вне зазора, когда труба близко
                reward -= 5
        
        # Небольшой штраф за нахождение слишком близко к земле или потолку, независимо от труб
        if bird_y_norm < 0.1 or bird_y_norm > 0.9: # 10% от верха/низа экрана
            reward -= 2
        elif bird_y_norm < 0.15 or bird_y_norm > 0.85: # 15% от верха/низа экрана
            reward -= 1
            
        return reward
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval() # Переключаем в режим оценки
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train() # Возвращаем в режим обучения
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0 # Возвращаем 0 для индикации, что обучение не произошло
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).unsqueeze(1).to(self.device) # unsqueeze для gather
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values_target = self.target_network(next_states).max(1)[0]
            # Если следующий стейт терминальный, то его ценность 0
            next_q_values_target[dones] = 0.0 
        
        target_q_values = rewards + (0.99 * next_q_values_target) # gamma = 0.99
        
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values) # squeeze current_q_values
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item() # Возвращаем значение функции потерь
            
    def save(self, name):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, name)
        print(f"Модель сохранена в {name}")
        
    def load(self, name):
        # Проверяем, существует ли файл
        if not os.path.exists(name):
            print(f"Ошибка: Файл модели {name} не найден.")
            return False
        try:
            checkpoint = torch.load(name, map_location=self.device) # map_location для гибкости cpu/gpu
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['model_state_dict']) # Также загружаем в target_network
            if 'optimizer_state_dict' in checkpoint: # Для обратной совместимости, если ранее не сохраняли
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epsilon' in checkpoint:
                 self.epsilon = checkpoint['epsilon']
            print(f"Модель загружена из {name}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели {name}: {e}")
            return False


def train_smart_agent(episodes=3000, lr=0.00025, model_load_path=None):
    env = game.GameState()
    agent = SmartAgent(state_size=5, action_size=2, lr=lr)
    
    if model_load_path:
        if not agent.load(model_load_path):
            print(f"Не удалось загрузить модель {model_load_path}, начинаем обучение с нуля.")

    target_update_freq = 10  # Обновляем target network каждые 10 эпизодов (было 50, но можно чаще при коротких эпизодах)
    replay_start_size = 2000 # Начинаем обучение после N шагов в памяти (увеличено для большего исследования)
    batch_size = 64

    scores_history = []
    episode_rewards_history = []
    losses_history = []
    max_score_achieved = 0
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"Обучение SmartDQN агента на устройстве: {agent.device}")
    start_time = time.time()
    
    # Генерируем временную метку для названий файлов
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for episode in range(1, episodes + 1):
        env.__init__() # Сброс игры
        current_state_info = agent.get_enhanced_state(env)
        total_episode_reward = 0
        current_score = 0
        steps_in_episode = 0
        episode_loss_sum = 0
        episode_loss_count = 0
        
        while True:
            steps_in_episode += 1
            action = agent.act(current_state_info)
            
            action_input = [0, 0]
            if action == 1: # flap
                action_input[1] = 1
            else: # do nothing
                action_input[0] = 1
                
            _, env_reward, terminal = env.frame_step(action_input) # Изображение не используется напрямую агентом
            
            next_state_info = agent.get_enhanced_state(env)
            
            # Используем улучшенную систему награды агента
            # Передаем env, terminal, prev_score (current_score до обновления), next_state_info
            shaped_reward = agent.get_shaped_reward(env, terminal, current_score, next_state_info)
            total_episode_reward += shaped_reward # Суммируем "умные" награды
            
            agent.remember(current_state_info, action, shaped_reward, next_state_info, terminal)
            current_state_info = next_state_info
            current_score = env.score # Обновляем счет после шага
            
            if len(agent.memory) > replay_start_size:
                loss = agent.replay(batch_size)
                if loss > 0: # Если обучение произошло
                    episode_loss_sum += loss
                    episode_loss_count += 1
            
            if terminal or steps_in_episode > 7000: # Ограничение на шаги в эпизоде
                break
        
        scores_history.append(current_score)
        episode_rewards_history.append(total_episode_reward)
        if episode_loss_count > 0:
            losses_history.append(episode_loss_sum / episode_loss_count)
        else:
            losses_history.append(None) # Если не было обучения в этом эпизоде
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        if current_score > max_score_achieved:
            max_score_achieved = current_score
            # Добавляем временную метку к названию файла модели
            agent.save(f'models/smart_agent_best_score_{max_score_achieved}_{timestamp}.pt')
            print(f"🎉 Новый рекорд! Счет: {max_score_achieved} (эпизод {episode})")
        
        avg_score_last_100 = np.mean(scores_history[-100:]) if len(scores_history) >= 100 else np.mean(scores_history)
        avg_reward_last_100 = np.mean(episode_rewards_history[-100:]) if len(episode_rewards_history) >= 100 else np.mean(episode_rewards_history)

        # Логирование каждого эпизода
        print(f"Эп {episode:5d} | Счет: {current_score:3d} | Ср.счет(100): {avg_score_last_100:6.2f} | "
                  f"Макс: {max_score_achieved:3d} | Epsilon: {agent.epsilon:.4f} | "
                  f"Ср.награда(100): {avg_reward_last_100:8.2f} | Шаги: {steps_in_episode:4d}")
        
        if max_score_achieved >= 100: # Минимальное требование по ДЗ
            print(f"🏆 ЦЕЛЬ ДОСТИГНУТА! Максимальный счет: {max_score_achieved} в эпизоде {episode}.")
            # Добавляем временную метку к названию файла модели
            agent.save(f'models/smart_agent_target_reached_100_{timestamp}.pt')
            # Можно раскомментировать break, если нужно остановиться после достижения цели
            # break 
            
    # Финальное сохранение с временной меткой
    agent.save(f'models/smart_agent_final_{timestamp}.pt')
    total_training_time = time.time() - start_time
    print(f"Обучение завершено за {total_training_time/60:.2f} минут.")
    
    # Графики
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(scores_history)
    plt.title('Счет по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Счет')
    if len(scores_history) >= 100:
        moving_avg_scores = np.convolve(scores_history, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(scores_history)), moving_avg_scores, label='Ср. счет (100 эп.)', color='red')
        plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(episode_rewards_history)
    plt.title('Общая награда за эпизод')
    plt.xlabel('Эпизод')
    plt.ylabel('Общая награда')
    if len(episode_rewards_history) >= 100:
        moving_avg_rewards = np.convolve(episode_rewards_history, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards_history)), moving_avg_rewards, label='Ср. награда (100 эп.)', color='red')
        plt.legend()
        
    plt.subplot(2, 2, 3)
    valid_losses = [l for l in losses_history if l is not None]
    valid_indices = [i for i, l in enumerate(losses_history) if l is not None]
    if valid_losses:
        plt.plot(valid_indices, valid_losses)
        plt.title('Функция потерь (Loss) по эпизодам (где было обучение)')
        plt.xlabel('Эпизод')
        plt.ylabel('Loss')

    plt.subplot(2, 2, 4)
    epsilons = [agent.epsilon_min + (0.9 - agent.epsilon_min) * (agent.epsilon_decay**i) for i in range(episode)]
    if episode > 0 : # Проверяем, что был хотя бы один эпизод
        actual_epsilons_at_log_points = []
        initial_epsilon = 0.9 # или model_load_path если загружали
        current_epsilon_val = initial_epsilon
        # Приблизительное восстановление эпсилон, если обучение было не с нуля
        # Более точное восстановление сложно без сохранения истории эпсилон
        # Этот график будет скорее теоретическим убыванием эпсилон
        temp_eps = agent.epsilon if model_load_path else 0.9 
        eps_decay_rate = agent.epsilon_decay
        eps_min_val = agent.epsilon_min
        recorded_eps = []
        for i in range(1, episode + 1):
            recorded_eps.append(temp_eps)
            if len(agent.memory) > replay_start_size : # Эпсилон убывает только после начала replay
                 if temp_eps > eps_min_val:
                    temp_eps *= eps_decay_rate

        plt.plot(recorded_eps)
        plt.title('Epsilon Decay')
        plt.xlabel('Эпизод')
        plt.ylabel('Epsilon')
    
    plt.tight_layout()
    # Добавляем временную метку к названию файла графика
    plt.savefig(f'logs/smart_agent_training_report_{timestamp}.png', dpi=150)
    print(f"Графики обучения сохранены в logs/smart_agent_training_report_{timestamp}.png")
    # plt.show() # Раскомментируйте, если хотите показать графики сразу

    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ ОБУЧЕНИЯ SMARTDQN АГЕНТА")
    print(f"{'='*50}")
    print(f"Максимальный счет: {max_score_achieved}")
    print(f"Финальный средний счет (100 эпизодов): {avg_score_last_100:.2f}")
    print(f"Общее количество эпизодов: {episode}")
    print(f"Финальный epsilon: {agent.epsilon:.4f}")
    
    return agent, scores_history


def test_smart_agent(model_path, num_games=10):
    agent = SmartAgent(state_size=5, action_size=2)
    if not agent.load(model_path):
        print(f"Не удалось загрузить модель {model_path} для тестирования.")
        return []
        
    agent.epsilon = 0  # Отключаем exploration для тестирования
    
    env = game.GameState()
    scores = []
    
    print(f"\nТестируем модель: {model_path}")
    
    for game_num in range(1, num_games + 1):
        env.__init__()
        state = agent.get_enhanced_state(env)
        steps = 0
        current_game_score = 0
        
        while True:
            steps += 1
            action = agent.act(state)
            
            action_input = [0,0]
            if action == 1: action_input[1] = 1
            else: action_input[0] = 1
            
            _, _, terminal = env.frame_step(action_input) # Изображение и награда среды не нужны для теста
            state = agent.get_enhanced_state(env)
            current_game_score = env.score
            
            if terminal or steps > 10000: # Ограничение на шаги в игре
                break
        
        scores.append(current_game_score)
        print(f"Игра {game_num:2d}: Счет = {current_game_score:3d}, Шагов = {steps:4d}")
    
    avg_score = np.mean(scores) if scores else 0
    max_s = max(scores) if scores else 0
    min_s = min(scores) if scores else 0
    
    print(f"\n{'='*40}")
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({num_games} игр)")
    print(f"{'='*40}")
    print(f"Средний счет: {avg_score:.2f}")
    print(f"Максимальный счет: {max_s}")
    print(f"Минимальный счет: {min_s}")
    if scores:
        print(f"Процент игр с счетом >= 10: {sum(1 for s in scores if s >= 10) / len(scores) * 100:.1f}%")
        print(f"Процент игр с счетом >= 100: {sum(1 for s in scores if s >= 100) / len(scores) * 100:.1f}%")
    print(f"Все счета: {scores}")
    
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SmartDQN агент для Flappy Bird')
    parser.add_argument('--train', action='store_true', help='Обучить нового агента')
    parser.add_argument('--test', type=str, metavar='MODEL_PATH', help='Протестировать обученную модель (укажите путь к .pt файлу)')
    parser.add_argument('--load_and_train', type=str, metavar='MODEL_PATH', help='Загрузить модель и продолжить обучение')
    parser.add_argument('--episodes', type=int, default=3000, help='Количество эпизодов для обучения (по умолчанию: 3000)')
    parser.add_argument('--lr', type=float, default=0.00025, help='Скорость обучения (learning rate) (по умолчанию: 0.00025)')
    parser.add_argument('--games', type=int, default=10, help='Количество игр для тестирования (по умолчанию: 10)')
    
    args = parser.parse_args()
    
    # Убедимся, что папка game с wrapped_flappy_bird.py находится в PYTHONPATH или в той же директории
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Пример, если скрипт в подпапке
    
    if args.train:
        print("Начинаем обучение нового агента...")
        train_smart_agent(episodes=args.episodes, lr=args.lr)
    elif args.load_and_train:
        print(f"Загружаем модель из {args.load_and_train} и продолжаем обучение...")
        train_smart_agent(episodes=args.episodes, lr=args.lr, model_load_path=args.load_and_train)
    elif args.test:
        test_smart_agent(model_path=args.test, num_games=args.games)
    else:
        print("Не указан режим работы. Используйте --train для обучения нового агента, "
              "--load_and_train <path> для продолжения обучения, "
              "или --test <path> для тестирования существующей модели.")
        parser.print_help()