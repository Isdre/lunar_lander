import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# 1. Tworzenie środowiska LunarLander-v3
# Możesz użyć make_vec_env dla lepszej wydajności trenowania, nawet z jednym środowiskiem
vec_env = make_vec_env("LunarLander-v3", n_envs=1)

# 2. Inicjalizacja modelu A2C
# Polityka "MlpPolicy" jest odpowiednia dla środowisk z ciągłymi stanami
# i dyskretnymi akcjami, gdzie sieć neuronowa (MLP) uczy się mapowania
# stanu na rozkład prawdopodobieństwa akcji.
model = A2C("MlpPolicy", vec_env, learning_rate=0.007, n_steps=5, gamma=0.99, verbose=1)

# 3. Trenowanie modelu
print("Rozpoczynam trenowanie modelu A2C...")
model.learn(total_timesteps=50000) # Ilość kroków trenowania, którą można zwiększyć

# 4. Zapisanie wytrenowanego modelu
model.save("a2c_lunarlander")
print("Model A2C wytrenowany i zapisany jako a2c_lunarlander.zip")

# 5. Wczytanie i testowanie wytrenowanego modelu
print("Testowanie wytrenowanego modelu...")
del model # Usuń model z pamięci, aby upewnić się, że wczytujemy nowy
model = A2C.load("a2c_lunarlander", env=vec_env)

# VecEnv.reset() zwraca tylko obserwację
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    
    # POPRAWKA: VecEnv.step() zwraca 4 wartości: obs, rewards, dones, infos
    # W VecEnv "dones" jest tablicą boolean True, jeśli środowisko jest zakończone LUB skrócone
    obs, rewards, dones, info = vec_env.step(action)
    
    # Sprawdź, czy którekolwiek środowisko jest zakończone
    if dones.any():
        # POPRAWKA: VecEnv.reset() zwraca tylko obserwację
        obs = vec_env.reset()

# Zamknięcie środowiska po zakończeniu testowania
vec_env.close()
print("Testowanie zakończone.")
