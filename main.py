import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
import time

# Utworzenie środowiska LunarLander-v3
# render_mode jest potrzebny, aby włączyć wizualizację
env = gym.make("LunarLander-v3", render_mode="human")

# Zresetowanie środowiska i uzyskanie początkowej obserwacji oraz informacji
obs, info = env.reset()

# Wczytanie wytrenowanego modelu A2C
# Upewnij się, że plik 'a2c_lunarlander.zip' znajduje się w tym samym katalogu
# lub podaj pełną ścieżkę do pliku.
# model = A2C.load("a2c_lunarlander", env=env)
model = PPO.load("./best_model/best_model", env=env)

# print("Testowanie wytrenowanego modelu A2C...")
print("Testowanie wytrenowanego modelu PPO...")
# Liczba epizodów do przetestowania
num_episodes = 5

for episode in range(num_episodes):
    # Zresetowanie środowiska na początku każdego epizodu
    obs, info = env.reset()
    done = False
    total_reward = 0

    print(f"Rozpoczynam epizod {episode + 1}/{num_episodes}")

    # Pętla testowa dla pojedynczego epizodu
    while not done:
        # Model przewiduje akcję na podstawie bieżącej obserwacji
        # deterministic=True oznacza, że zawsze wybieramy akcję o najwyższym prawdopodobieństwie
        action, _states = model.predict(obs, deterministic=True)

        # Wykonanie akcji w środowisku i uzyskanie nowej obserwacji, nagrody,
        # flag zakończenia (terminated/truncated) oraz informacji
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Obliczanie, czy epizod został zakończony (terminated) lub przerwany (truncated)
        done = terminated or truncated
        total_reward += reward

        # Renderowanie środowiska, aby wizualizować działanie lądownika
        env.render()
        
        # Opcjonalnie: krótka pauza, aby wizualizacja była płynniejsza
        time.sleep(0.01)

    print(f"Epizod {episode + 1} zakończony. Całkowita nagroda: {total_reward:.2f}")

# Zamknięcie środowiska po zakończeniu wszystkich epizodów testowych
env.close()
print("Testowanie modelu zakończone.")