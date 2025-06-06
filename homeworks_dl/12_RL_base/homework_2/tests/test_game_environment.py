import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Добавляем корневую папку проекта в Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_dqn_agent import SmartAgent


class TestGameEnvironment:
    """Тесты для игровой среды и вспомогательных функций"""
    
    def test_get_enhanced_state_function(self):
        """Тест функции get_enhanced_state"""
        agent = SmartAgent()
        
        # Мокаем объект env
        mock_env = Mock()
        mock_env.playerx = 100.0
        mock_env.playery = 200.0
        mock_env.playerVelY = -5.0
        mock_env.lowerPipes = [{'x': 150, 'y': 300}]
        mock_env.upperPipes = [{'x': 150, 'y': 200}]
        
        state = agent.get_enhanced_state(mock_env)
        
        assert isinstance(state, np.ndarray), "State должно быть numpy array"
        assert state.shape == (5,), "State должно иметь размер 5"
        assert len(state) == 5, "State должно содержать 5 элементов"
        
        # Проверяем, что значения в разумных пределах
        assert all(isinstance(x, (int, float, np.number)) for x in state), "Все элементы должны быть числами"
        
    def test_get_enhanced_state_with_multiple_pipes(self):
        """Тест get_enhanced_state с множественными трубами"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.playerx = 100.0
        mock_env.playery = 200.0
        mock_env.playerVelY = -5.0
        # Несколько труб - должна выбираться ближайшая справа
        mock_env.lowerPipes = [
            {'x': 80, 'y': 300},    # Слева от птицы
            {'x': 150, 'y': 300},   # Ближайшая справа
            {'x': 200, 'y': 300}    # Дальше справа
        ]
        mock_env.upperPipes = [
            {'x': 80, 'y': 200},
            {'x': 150, 'y': 200},
            {'x': 200, 'y': 200}
        ]
        
        state = agent.get_enhanced_state(mock_env)
        
        # Проверяем базовую структуру состояния
        assert len(state) == 5, "State должно содержать 5 элементов"
        assert all(isinstance(x, (int, float, np.number)) for x in state), "Все элементы должны быть числами"
        
    def test_get_enhanced_state_no_pipes_ahead(self):
        """Тест get_enhanced_state когда нет труб впереди"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.playerx = 200.0
        mock_env.playery = 200.0
        mock_env.playerVelY = -5.0
        # Все трубы позади птицы
        mock_env.lowerPipes = [{'x': 150, 'y': 300}]
        mock_env.upperPipes = [{'x': 150, 'y': 200}]
        
        state = agent.get_enhanced_state(mock_env)
        
        # Проверяем, что состояние создается корректно
        assert len(state) == 5, "State должно содержать 5 элементов"
        assert all(isinstance(x, (int, float, np.number)) for x in state), "Все элементы должны быть числами"
        
    def test_get_shaped_reward_survival(self):
        """Тест базовой награды за выживание"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.score = 0
        
        # Создаем mock состояние: птица далеко от трубы, чтобы не было позиционных штрафов
        # [bird_y_norm=0.5, bird_vel, next_pipe_dist_norm=0.8 (далеко), gap_center=0.5, gap_top=0.3]
        state_info = np.array([0.5, -2, 0.8, 0.5, 0.3])
        
        reward = agent.get_shaped_reward(mock_env, False, 0, state_info)
        
        # Должна быть базовая награда за выживание (1.0), без штрафов
        assert reward == 1.0, f"Должна быть базовая награда за выживание 1.0, получено {reward}"
        
    def test_get_shaped_reward_death(self):
        """Тест наказания за смерть"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.score = 0
        
        state_info = np.array([100, 0, -2, 250, 100])
        
        reward = agent.get_shaped_reward(mock_env, True, 0, state_info)
        
        # Должно быть большое наказание за смерть
        assert reward == -100, f"Наказание за смерть должно быть -100, получено {reward}"
        
    def test_get_shaped_reward_score_increase(self):
        """Тест награды за увеличение счета"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.score = 1  # Новый счет больше предыдущего
        
        state_info = np.array([100, 0, -2, 250, 100])
        
        reward = agent.get_shaped_reward(mock_env, False, 0, state_info)  # prev_score = 0
        
        # Должна быть большая награда за прохождение трубы
        assert reward == 200, f"Награда за прохождение трубы должна быть 200, получено {reward}"
        
    def test_get_shaped_reward_position_based(self):
        """Тест позиционных наград"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.score = 0
        
        # Птица близко к трубе и хорошо позиционирована в центре зазора
        # bird_y_norm в центре зазора, труба близко
        gap_center = 0.5
        gap_top = 0.4  # Верх зазора
        bird_y = gap_center  # Птица точно в центре зазора
        state_info = np.array([bird_y, -2, 0.2, gap_center, gap_top])  # близко к трубе
        
        reward = agent.get_shaped_reward(mock_env, False, 0, state_info)
        
        # Должна быть положительная награда (выживание + позиционная)
        assert reward > 1.0, f"Должна быть дополнительная награда за хорошее позиционирование, получено {reward}"
        
    def test_get_shaped_reward_penalty_for_bad_position(self):
        """Тест штрафа за плохое позиционирование"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.score = 0
        
        # Птица далеко от центра зазора
        state_info = np.array([50, 60, -2, 250, 100])  # далеко от центра
        
        reward = agent.get_shaped_reward(mock_env, False, 0, state_info)
        
        # Должен быть штраф за плохое позиционирование
        assert reward < 1.0, "Должен быть штраф за плохое позиционирование"
        
    def test_state_consistency(self):
        """Тест консистентности создания состояний"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.playerx = 100.0
        mock_env.playery = 200.0
        mock_env.playerVelY = -5.0
        mock_env.lowerPipes = [{'x': 150, 'y': 300}]
        mock_env.upperPipes = [{'x': 150, 'y': 200}]
        
        # Создаем состояние несколько раз
        state1 = agent.get_enhanced_state(mock_env)
        state2 = agent.get_enhanced_state(mock_env)
        
        # Состояния должны быть идентичными при одинаковых входных данных
        assert np.array_equal(state1, state2), "Состояния должны быть консистентными"
        
    def test_state_value_ranges(self):
        """Тест разумности диапазонов значений состояния"""
        agent = SmartAgent()
        
        mock_env = Mock()
        mock_env.playerx = 100.0
        mock_env.playery = 200.0
        mock_env.playerVelY = -5.0
        mock_env.lowerPipes = [{'x': 150, 'y': 300}]
        mock_env.upperPipes = [{'x': 150, 'y': 200}]
        
        state = agent.get_enhanced_state(mock_env)
        
        # Проверяем разумность значений
        horizontal_distance = state[0]
        vertical_distance = state[1]
        player_velocity = state[2]
        gap_position = state[3]
        gap_size = state[4]
        
        assert abs(vertical_distance) < 1000, "Вертикальное расстояние должно быть в разумных пределах"
        assert abs(player_velocity) < 100, "Скорость игрока должна быть в разумных пределах"
        assert gap_position >= 0, "Позиция зазора не должна быть отрицательной"
        assert gap_size > 0, "Размер зазора должен быть положительным" 