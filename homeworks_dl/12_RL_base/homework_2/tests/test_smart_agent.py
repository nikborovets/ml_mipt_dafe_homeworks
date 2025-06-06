import pytest
import torch
import numpy as np
import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Добавляем корневую папку проекта в Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_dqn_agent import SmartAgent


class TestSmartAgent:
    """Тесты для класса SmartAgent"""
    
    def test_smart_agent_initialization(self):
        """Тест инициализации SmartAgent с параметрами по умолчанию"""
        agent = SmartAgent()
        
        assert agent.state_size == 5
        assert agent.action_size == 2
        assert agent.epsilon == 0.9
        assert agent.epsilon_min == 0.001
        assert agent.epsilon_decay == 0.9999
        assert len(agent.memory) == 0
        assert hasattr(agent, 'q_network')
        assert hasattr(agent, 'target_network')
        assert hasattr(agent, 'optimizer')
        
    def test_smart_agent_custom_parameters(self):
        """Тест инициализации SmartAgent с кастомными параметрами"""
        agent = SmartAgent(state_size=10, action_size=4, lr=0.001)
        
        assert agent.state_size == 10
        assert agent.action_size == 4
        assert agent.learning_rate == 0.001
        
    def test_remember_method(self):
        """Тест метода remember для добавления опыта в память"""
        agent = SmartAgent()
        
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        action = 1
        reward = 10.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        done = False
        
        agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 1
        stored_experience = agent.memory[0]
        assert np.array_equal(stored_experience[0], state)
        assert stored_experience[1] == action
        assert stored_experience[2] == reward
        assert np.array_equal(stored_experience[3], next_state)
        assert stored_experience[4] == done
        
    def test_act_method_deterministic(self):
        """Тест метода act в детерминистическом режиме (epsilon=0)"""
        agent = SmartAgent()
        agent.epsilon = 0  # Отключаем случайность
        
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        action = agent.act(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < agent.action_size
        
    def test_act_method_random(self):
        """Тест метода act в случайном режиме (epsilon=1)"""
        agent = SmartAgent()
        agent.epsilon = 1.0  # Включаем полную случайность
        
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Запускаем несколько раз, чтобы проверить случайность
        actions = [agent.act(state) for _ in range(20)]
        
        # Проверяем, что все действия в корректном диапазоне
        for action in actions:
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < agent.action_size
            
        # При полной случайности должно быть разнообразие действий
        unique_actions = set(actions)
        assert len(unique_actions) > 1, "При epsilon=1 должны быть разные действия"
        
    def test_update_target_network(self):
        """Тест обновления target network"""
        agent = SmartAgent()
        
        # Изменяем веса основной сети
        with torch.no_grad():
            for param in agent.q_network.parameters():
                param.fill_(1.0)
                
        # Обновляем target network
        agent.update_target_network()
        
        # Проверяем, что веса скопированы
        for q_param, target_param in zip(agent.q_network.parameters(), agent.target_network.parameters()):
            assert torch.allclose(q_param, target_param), "Target network не обновлен корректно"
            
    def test_replay_insufficient_memory(self):
        """Тест replay при недостаточном количестве опыта в памяти"""
        agent = SmartAgent()
        
        # Добавляем мало опыта
        for i in range(30):  # Меньше чем batch_size=64
            state = np.random.randn(5)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = False
            agent.remember(state, action, reward, next_state, done)
            
        # replay должен вернуть 0 (обучение не произошло)
        loss = agent.replay(batch_size=64)
        assert loss == 0, "При недостаточной памяти replay должен вернуть 0"
        
    def test_replay_sufficient_memory(self):
        """Тест replay при достаточном количестве опыта в памяти"""
        agent = SmartAgent()
        
        # Добавляем достаточно опыта
        for i in range(100):
            state = np.random.randn(5)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = np.random.choice([True, False])
            agent.remember(state, action, reward, next_state, done)
            
        # replay должен вернуть значение loss > 0
        loss = agent.replay(batch_size=32)
        assert isinstance(loss, float), "replay должен вернуть float значение loss"
        assert loss >= 0, "Loss должен быть неотрицательным"
        
    def test_epsilon_decay(self):
        """Тест затухания epsilon"""
        agent = SmartAgent()
        initial_epsilon = agent.epsilon
        
        # Добавляем опыт и запускаем replay несколько раз
        for i in range(100):
            state = np.random.randn(5)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = False
            agent.remember(state, action, reward, next_state, done)
            
        # Запускаем replay несколько раз
        for _ in range(10):
            agent.replay(batch_size=32)
            
        # Epsilon должен уменьшиться, но не ниже минимума
        assert agent.epsilon <= initial_epsilon, "Epsilon должен уменьшиться"
        assert agent.epsilon >= agent.epsilon_min, "Epsilon не должен быть ниже минимума"
        
    def test_save_and_load_model(self):
        """Тест сохранения и загрузки модели"""
        agent1 = SmartAgent()
        
        # Устанавливаем определенные значения
        agent1.epsilon = 0.5
        with torch.no_grad():
            for param in agent1.q_network.parameters():
                param.fill_(2.0)
                
        # Сохраняем в временный файл
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            agent1.save(temp_path)
            
            # Создаем новый агент и загружаем модель
            agent2 = SmartAgent()
            success = agent2.load(temp_path)
            
            assert success, "Загрузка модели должна быть успешной"
            assert agent2.epsilon == 0.5, "Epsilon должен быть загружен корректно"
            
            # Проверяем, что веса сети загружены
            for param1, param2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
                assert torch.allclose(param1, param2), "Веса сети должны быть идентичными"
                
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_load_nonexistent_model(self):
        """Тест загрузки несуществующей модели"""
        agent = SmartAgent()
        
        success = agent.load("nonexistent_file.pt")
        assert not success, "Загрузка несуществующего файла должна вернуть False" 