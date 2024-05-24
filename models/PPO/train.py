
# pip install 'stable-baselines3==1.7.0' --force-reinstall
# pip install numpy gym stable-baselines3

import json
import numpy as np
import os
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class HalsteadEnv(Env):
    def __init__(self, data):
        super(HalsteadEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Ação: escolher um provedor (0, 1, 2 ou 3)
        self.action_space = Discrete(4)

        # Observações: métricas de Halstead
        self.observation_space = Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step]
        metrics = [
            obs["length"], 
            obs["vocabulary"], 
            obs["difficulty"], 
            obs["volume"],
            obs["effort"], 
            obs["bugs"], 
            obs["time"], 
            obs["distinct_operators"],
            obs["total_operators"], 
            obs["distinct_operands"], 
            obs["total_operands"],
            obs["provider"]
        ]
        return np.array(metrics, dtype=np.float32)

    def step(self, action):
        obs = self.data[self.current_step]
        #reward = 1.0 if action - (action * 0.2)  <= obs["execution_time"] and action + (action * 0.2) > obs["execution_time"] else -1.0
        reward = 1 + obs["execution_time"] - action
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            return np.zeros(self.observation_space.shape), reward, done, {}

        return self._next_observation(), reward, done, {}

def load_data_from_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename)) as f:
                data.extend(json.load(f))
    return data

if __name__ == "__main__":
    # Carregar dados dos arquivos JSON
    directory = 'samples'
    data = load_data_from_json_files(directory)

    # Inicializar o ambiente com os dados
    env = HalsteadEnv(data)

    # Verificar o ambiente (opcional, mas recomendado)
    check_env(env)

    # Treinar o agente usando PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Salvar o modelo treinado
    model.save("halstead_ppo_model")

    # Avaliar o agente
    obs = env.reset()
    total_reward = 0
    total_error = 0
    for _ in range(len(data)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs[11])
        print(reward)
        total_error += obs[11] - reward
        total_reward += reward
        if done:
            break

    print("Recompensa total após a avaliação:", total_reward)
    print("Erro total após a avaliação:", total_error)

