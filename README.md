# 🧠 Tik-Tac-Toe-Reinforcement-Learner

This project implements a **reinforcement learning (RL) agent** that learns to play Tic-Tac-Toe using a **Monte Carlo control approach with Q-learning updates**.
The agent improves through **self-play vs a random opponent**, updating its Q-table after each episode.

---

## 📖 Approach

* State representation: **board encoded as tuple of tuples**
* Action selection: **epsilon-greedy policy**
* Update rule:

```python
# Monte Carlo Q-update
Q[s][a] = Q[s][a] + alpha * (G_t - Q[s][a])
```

where `G_t` is the discounted return from step `t`.

* Rewards:

  * `+1` → win
  * `-1` → loss
  * `0.3` → draw
  * `0` → non-terminal moves

---

## 📊 Results

Training curve (win rate vs episodes):

<img width="1157" height="478" alt="Screenshot 2025-09-13 at 3 02 03 PM" src="https://github.com/user-attachments/assets/dda807b4-9808-4a1d-9021-386c3457fb65" />


Example game played by the AI agent:

<img width="238" height="492" alt="Screenshot 2025-09-13 at 2 44 35 PM" src="https://github.com/user-attachments/assets/61835690-8014-473b-81e9-19fd7f8b7a4f" />


The AI starts with random moves, but over time develops a strategy that wins consistently against a random opponent and can also challenge human players.
