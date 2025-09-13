import random
from abc import ABC, abstractmethod
from typing import Tuple,List,Optional,Literal,Dict,cast
from dataclasses import dataclass

TikTakToeLocalState=Tuple[
    Tuple[Literal['.','+','*'],Literal['.','+','*'],Literal['.','+','*']],
    Tuple[Literal['.','+','*'],Literal['.','+','*'],Literal['.','+','*']],
    Tuple[Literal['.','+','*'],Literal['.','+','*'],Literal['.','+','*']]
]

TikTakToeGlobalState=Tuple[
    Tuple[Literal['x','o','_'],Literal['x','o','_'],Literal['x','o','_']],
    Tuple[Literal['x','o','_'],Literal['x','o','_'],Literal['x','o','_']],
    Tuple[Literal['x','o','_'],Literal['x','o','_'],Literal['x','o','_']]
]
class Player(ABC):
    @abstractmethod
    def get_pos(self, state:TikTakToeLocalState)->Tuple[int,int]:
        pass
    @abstractmethod
    def on_victory(self,state:TikTakToeLocalState)->None:
        pass
    @abstractmethod
    def on_draw(self,state:TikTakToeLocalState)->None:
        pass
    @abstractmethod
    def on_defeat(self,state:TikTakToeLocalState)->None:
        pass
    @abstractmethod
    def on_progress(self,state:TikTakToeLocalState)->None:
        pass
    @abstractmethod
    def greet(self,result:Literal['win','lose','draw']):
        pass

@dataclass
class Result:
    winner:Optional[Literal['x','o']]
    loser:Optional[Literal['x','o']]
    draw:bool
    winning_greet_state:Optional[List[List[Literal['x','o','❌','🅾️','_']]]]

def get_result(state: TikTakToeGlobalState) -> Result:
    # create a full copy (keep '_' so indices match original)
    greet_state: List[List[Literal['x','o','❌','🅾️','_']]] = [[el for el in row] for row in state]

    # check rows
    for i in range(3):
        if all(slot == 'x' for slot in state[i]):
            for j in range(3):
                greet_state[i][j] = '❌'
            return Result(winner='x', loser='o', draw=False, winning_greet_state=greet_state)
        if all(slot == 'o' for slot in state[i]):
            for j in range(3):
                greet_state[i][j] = '🅾️'
            return Result(winner='o', loser='x', draw=False, winning_greet_state=greet_state)

    # check columns
    for j in range(3):
        if all(state[i][j] == 'x' for i in range(3)):
            for i in range(3):
                greet_state[i][j] = '❌'
            return Result(winner='x', loser='o', draw=False, winning_greet_state=greet_state)
        if all(state[i][j] == 'o' for i in range(3)):
            for i in range(3):
                greet_state[i][j] = '🅾️'
            return Result(winner='o', loser='x', draw=False, winning_greet_state=greet_state)

    # diagonals
    if all(state[i][i] == 'x' for i in range(3)):
        for i in range(3):
            greet_state[i][i] = '❌'
        return Result(winner='x', loser='o', draw=False, winning_greet_state=greet_state)
    if all(state[i][i] == 'o' for i in range(3)):
        for i in range(3):
            greet_state[i][i] = '🅾️'
        return Result(winner='o', loser='x', draw=False, winning_greet_state=greet_state)

    if all(state[i][2 - i] == 'x' for i in range(3)):
        for i in range(3):
            greet_state[i][2 - i] = '❌'
        return Result(winner='x', loser='o', draw=False, winning_greet_state=greet_state)
    if all(state[i][2 - i] == 'o' for i in range(3)):
        for i in range(3):
            greet_state[i][2 - i] = '🅾️'
        return Result(winner='o', loser='x', draw=False, winning_greet_state=greet_state)

    # if no winner, check if there are still empty slots -> not finished; else draw
    any_empty = any(cell == '_' for row in state for cell in row)
    if any_empty:
        return Result(winner=None, loser=None, draw=False, winning_greet_state=None)
    else:
        return Result(winner=None, loser=None, draw=True, winning_greet_state=None)

def get_local_game_state(state:TikTakToeGlobalState, player_symbol:Literal['x','o'])->TikTakToeLocalState:
    result= tuple(
        tuple(
            (
                '*' if el==player_symbol
                else '+' if el!=player_symbol and el!='_'
                else '.'
            )
            for el in row
        ) 
        for row in state
    )# * means current player, + means other player, . empty space
    return cast(TikTakToeLocalState,result)
def update_state(grid: TikTakToeGlobalState, row: int, col: int, value:Literal['x','o']) -> TikTakToeGlobalState:
    # Convert the row to a list so we can modify it
    new_row = list(grid[row])
    new_row[col] = value

    # Rebuild the grid with the updated row
    new_grid = grid[:row] + (tuple(new_row),) + grid[row+1:]
    return cast(TikTakToeGlobalState,new_grid)

def start_game(player_x:Player,player_o:Player, verbose:bool=False):
    state:TikTakToeGlobalState=(('_','_','_'),('_','_','_'),('_','_','_'))
    symbol_to_player:Dict[Literal['x','o'],Player]={'x':player_x, 'o':player_o}
    current_player_index=0
    while True:
        current_player_symbol:Literal['x','o']=list(symbol_to_player.keys())[current_player_index]
        player_local_current_state=get_local_game_state(state=state, player_symbol=current_player_symbol)

        if verbose:
            print(f"waiting for {current_player_symbol}...")
        r,c=symbol_to_player[current_player_symbol].get_pos(player_local_current_state)

        if 0<=r<3 and 0<=c<3 and state[r][c]=='_':
            state=update_state(state,r,c,current_player_symbol)
            if verbose:
                for row in state:
                    print(' '.join(row))
            player_local_nxt_state=get_local_game_state(state=state, player_symbol=current_player_symbol)
            result=get_result(state)
            winner=result.winner
            loser=result.loser
            is_draw=result.draw
            greet_matrix=result.winning_greet_state
            if is_draw:
                symbol_to_player['x'].on_draw(player_local_nxt_state)
                symbol_to_player['o'].on_draw(player_local_nxt_state)
                if verbose:
                    symbol_to_player['x'].greet("draw")
                    symbol_to_player['o'].greet("draw")
                return
            elif not winner and not loser:
                symbol_to_player['x'].on_progress(player_local_nxt_state)
                symbol_to_player['o'].on_progress(player_local_nxt_state)
            elif winner and loser and greet_matrix:   
                symbol_to_player[winner].on_victory(player_local_nxt_state)
                symbol_to_player[loser].on_defeat(player_local_nxt_state)
                if verbose:
                    symbol_to_player[winner].greet("win")
                    for row in greet_matrix:
                        print(' '.join(row))
                    symbol_to_player[loser].greet("lose")
                return
            current_player_index+=1
            current_player_index%=2
        else:
            print("invalid location ❌")

class CliPlayer(Player):
    def get_pos(self, state: TikTakToeLocalState) -> Tuple[int,int]:
        r=int(input("input row: "))
        c=int(input("input column: "))
        return r,c
    def on_victory(self, state: TikTakToeLocalState)->None:
        return
    def on_progress(self, state: TikTakToeLocalState) -> None:
        return
    def on_defeat(self, state: TikTakToeLocalState) -> None:
        return
    def on_draw(self, state: TikTakToeLocalState) -> None:
        return
    def greet(self, result: Literal['win','lose','draw']):
        if result=='win':
            print("🥳🎉🥂 congrats to user 🧑🏻‍🏫")
        elif result=='lose':
            print("Sorry user 🧑🏻‍🏫 😞")
        else:
            print("Draw 😂")


from collections import defaultdict
import statistics

# reuse your type alias names
# TikTakToeLocalState and TikTakToeGlobalState are defined by you externally

class RLPlayer(Player):
    @dataclass
    class HistoryItem:
        prev_state: TikTakToeLocalState
        action: Tuple[int, int]
        reward: float

    def __init__(self, q_table: Dict[str, Dict[str, float]] | None = None) -> None:
        # Q-table and hyperparams
        self.Q_table: Dict[str, Dict[str, float]] = {} if q_table is None else q_table
        self.q_update_parameter: float = 0.1    # alpha
        self.reward_discount: float = 1.0       # gamma

        # exploration
        self.exploration_prob: float = 0.1      # epsilon (epsilon-greedy)
        self.training: bool = False

        # per-episode temporary storage
        self.history: List[RLPlayer.HistoryItem] = []
        self.prev_state: Optional[TikTakToeLocalState] = None
        self.action_taken: Optional[Tuple[int, int]] = None

        # batch training settings
        self.training_batch_size: int = 1     # <-- configurable batch size
        self._batch_count: int = 0
        self._batch_histories: List[List[RLPlayer.HistoryItem]] = []  # list of episode-histories

        # diagnostics
        self.win_history: List[bool] = []

    # -------------------------
    # helpers
    # -------------------------
    def _state_to_key(self, state: TikTakToeLocalState) -> str:
        # tuple of tuples -> deterministic string key
        return ''.join(''.join(row) for row in state)

    def _action_to_key(self, action: Tuple[int, int]) -> str:
        return f"{action[0]}{action[1]}"

    def _append_current_move(self, reward: float) -> None:
        """
        Record the last (prev_state, action_taken, reward) as a HistoryItem.
        Because TikTakToeLocalState is a tuple-of-tuples (immutable), it is safe to store as-is.
        """
        if self.prev_state is None or self.action_taken is None:
            return
        self.history.append(RLPlayer.HistoryItem(
            prev_state=self.prev_state,
            action=self.action_taken,
            reward=reward
        ))
        # clear recorded pair to be ready for next get_pos
        self.prev_state = None
        self.action_taken = None

    def _compute_returns_for_episode(self, episode_history: List[HistoryItem]) -> List[float]:
        """
        Return a list of returns [G_0, G_1, ..., G_{T-1}] for that episode.
        G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_k
        """
        T = len(episode_history)
        returns: List[float] = [0.0] * T
        for t in range(T):
            G = 0.0
            for k in range(t, T):
                G += (self.reward_discount ** (k - t)) * episode_history[k].reward
            returns[t] = G
        return returns

    # -------------------------
    # batch update logic
    # -------------------------
    def _accumulate_episode_and_maybe_update(self) -> None:
        """
        Add current episode's history to the batch. If batch is full, call batch updater.
        """
        if self.history:
            # HistoryItem.prev_state is immutable (tuple) so shallow copy of list is enough
            episode_copy = list(self.history)
            self._batch_histories.append(episode_copy)
            self._batch_count += 1

        # clear per-episode history (we will update per-batch instead)
        self.history = []

        if self._batch_count >= self.training_batch_size:
            self.update_Q_table_batch()
            # reset batch accumulators
            self._batch_histories = []
            self._batch_count = 0

    def update_Q_table_batch(self) -> None:
        """
        For each (state,action) seen across the batch of episodes:
          - compute returns G_t for every visit (every-visit MC),
          - average G_t across all visits in the entire batch,
          - update Q(s,a) <- Q(s,a) + alpha * (avg_G - Q(s,a))
        """
        returns_by_sa: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        # compute returns for each episode and accumulate returns for each (s,a)
        for episode in self._batch_histories:
            if not episode:
                continue
            episode_returns = self._compute_returns_for_episode(episode)
            for item, G_t in zip(episode, episode_returns):
                s_key = self._state_to_key(item.prev_state)
                a_key = self._action_to_key(item.action)
                returns_by_sa[(s_key, a_key)].append(G_t)

        # average returns and update Q table
        for (s_key, a_key), g_list in returns_by_sa.items():
            avg_G = statistics.mean(g_list)
            if s_key not in self.Q_table:
                self.Q_table[s_key] = {}
            q_old = self.Q_table[s_key].get(a_key, 0.0)
            self.Q_table[s_key][a_key] = q_old + self.q_update_parameter * (avg_G - q_old)

    # -------------------------
    # policy / action selection
    # -------------------------
    def get_pos(self, state: TikTakToeLocalState) -> Tuple[int, int]:
        # store current state (tuple is immutable so safe)
        self.prev_state = state
        state_key = self._state_to_key(state)

        available: List[Tuple[float, Tuple[int, int]]] = []
        for i, row in enumerate(state):
            for j, el in enumerate(row):
                if el == '.':
                    a_key = f"{i}{j}"
                    qv = self.Q_table.get(state_key, {}).get(a_key, 0.0)
                    available.append((qv, (i, j)))

        if not available:
            return (0, 0)

        # choose max-value actions, break ties randomly
        best_value = max(v for v, _ in available)
        best_actions = [a for v, a in available if v == best_value]
        chosen_action = random.choice(best_actions)

        # epsilon-greedy exploration
        if self.training and random.random() < self.exploration_prob:
            chosen_action = random.choice([a for _, a in available])

        self.action_taken = chosen_action
        return chosen_action

    # -------------------------
    # callbacks (episode end / progress)
    # -------------------------
    def on_progress(self, state: TikTakToeLocalState) -> None:
        if not self.training:
            self.prev_state = None
            self.action_taken = None
            return
        # append last action with reward 0
        self._append_current_move(reward=0.0)

    def on_victory(self, state: TikTakToeLocalState) -> None:
        if not self.training:
            return
        self._append_current_move(reward=1.0)
        self.win_history.append(True)
        self._accumulate_episode_and_maybe_update()

    def on_defeat(self, state: TikTakToeLocalState) -> None:
        if not self.training:
            return
        self._append_current_move(reward=-1.0)
        self.win_history.append(False)
        self._accumulate_episode_and_maybe_update()

    def on_draw(self, state: TikTakToeLocalState) -> None:
        if not self.training:
            return
        # you used 0.3 previously for draw reward, preserved here
        self._append_current_move(reward=0.3)
        self.win_history.append(False)
        self._accumulate_episode_and_maybe_update()

    def greet(self, result: Literal['win', 'lose', 'draw']):
        if result == 'win':
            print("🥳🎉🥂 congrats to AI brain 🧠")
        elif result == 'lose':
            print("Sorry AI brain 🧠 😞")
        else:
            print("Draw 😂")



class RandomMadPlayer(Player):
    def get_pos(self, state: TikTakToeLocalState) -> Tuple[int,int]:
        available_slots=[]
        for i,row in enumerate(state):
            for j,el in enumerate(row):
                if el=='.':
                    available_slots.append((i,j))
        random_slot_index=random.randint(0,len(available_slots)-1)
        return available_slots[random_slot_index]
    def on_victory(self, state: TikTakToeLocalState)->None:
        return
    def on_progress(self, state: TikTakToeLocalState) -> None:
        return
    def on_defeat(self, state: TikTakToeLocalState) -> None:
        return
    def on_draw(self, state: TikTakToeLocalState) -> None:
        return
    def greet(self, result: Literal['win','lose','draw']):
        if result=='win':
            print("🥳🎉🥂 congrats to bot 🤖")
        elif result=='lose':
            print("Sorry bot 🤖 😞")
        else:
            print("Draw 😂")


import pickle
import time
from pathlib import Path
if __name__=="__main__":
    q_table_cache=Path("q_table_trained.pkl")
    win_history_cache=Path("win_history.pkl")
    # -------------------------
    # Train
    # -------------------------
    # rl_player=RLPlayer(q_table=None)
    # rl_player.training=True
    # rl_player.training_batch_size=100
    # epoch=500000
    # progress_segment_count=50
    # progress_segment_size=epoch//progress_segment_count
    # for i in range(epoch):
    #     order1=random.randint(0,1)
    #     if order1:
    #         start_game(RandomMadPlayer(), rl_player)
    #     else:
    #         start_game(rl_player,RandomMadPlayer())

    #     n_seg=(i+1)//progress_segment_size
    #     progress_bar=''.join(["#" for _ in range(n_seg)])
    #     print(f"{progress_bar} |{(i+1)*100/epoch:.2f}%",end="\r")
        
    # # Save Q table to disc

    # with open(q_table_cache,'wb') as f:
    #     pickle.dump(rl_player.Q_table,f)
    #     print(f"✅ saved q-table as {f.name}")
    # # Save winning history to disc
    # with open(win_history_cache,'wb') as f:
    #     pickle.dump(rl_player.win_history,f)
    #     print(f"✅ saved history as {f.name}")


    # -------------------------
    # Plot Win Rate
    # -------------------------
    # with open(win_history_cache, "rb") as f:
    #     win_hist = pickle.load(f)
    # import matplotlib.pyplot as plt
    # import numpy as np
    # arr = np.array(win_hist, dtype=int)
    # # Bin size
    # bin_size = 1000
    # # Trim to fit exact bins
    # n_bins = len(arr) // bin_size
    # arr = arr[:n_bins * bin_size]
    # # Reshape into bins and sum along axis=1
    # binned_counts = arr.reshape(n_bins, bin_size).sum(axis=1)*100/bin_size
    # # Plot
    # plt.figure(figsize=(12,5))
    # plt.plot(binned_counts)
    # plt.ylim(top=100)
    # plt.xlabel("Time index")
    # plt.ylabel("Count of Win")
    # plt.grid(True)
    # plt.show()
    # -------------------------
    # Play
    # -------------------------
    #Load the trained q table from cache
    if Path.exists(q_table_cache):
        with open(q_table_cache, "rb") as f:
            q_table = pickle.load(f)
    #play yourself as the starter
    start_game(CliPlayer(),RLPlayer(q_table=q_table), verbose=True)
    #play yourself as second player
    # start_game(CliPlayer(),RLPlayer(q_table=q_table), verbose=True)
    