class PluribusGame(gym.Env):
    def __init__(self):
        # Define action space
        self.action_space = spaces.Discrete(3)  # Fold, Call, Raise

        # Define observation space
        self.observation_space = spaces.Dict({
            "hand": spaces.Tuple((spaces.Discrete(52), spaces.Discrete(52))),
            "board": spaces.Tuple((spaces.Discrete(52), spaces.Discrete(52), spaces.Discrete(52), spaces.Discrete(52),
                                   spaces.Discrete(52))),
            "position": spaces.Discrete(6),
            "stack": spaces.Box(low=0, high=10000, shape=(1,), dtype=int),
            "current_bet": spaces.Box(low=0, high=10000, shape=(1,), dtype=int),
            "last_action": spaces.Discrete(3),
            "last_bet_size": spaces.Box(low=0, high=10000, shape=(1,), dtype=int),
            "current_player": spaces.Discrete(6),
        })

        # Initialize game state
        self.players = [Player(i, 10000) for i in range(6)]
        self.button = 0
        self.current_round = 0
        self.game_over = False
        self.deck = []
        self.board = []
        self.current_bet = 0
        self.last_raise = 100
        self.last_to_act = None
        self.pot = 0
        self.num_players = 6
        self.starting_stack = 10000
        self.reset()

    def reset(self):
        self.current_round = 0
        self.button = (self.button + 1) % self.num_players
        self.deck = self.create_deck()
        self.deal_hands()
        self.pot = 0
        self.current_bet = 0
        self.last_raise = 100
        self.last_to_act = None
        self.board = []
        self.game_over = False
        return self.get_observation()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        player = self.players[self.button]
        self.last_to_act = player
        if action == 0:
            self.players[self.button].stack += self.pot
            self.pot = 0
            self.reset()
            return self.get_observation(), 0, self.game_over, {}
        elif action == 1:
            self.players[self.button].stack -= (self.current_bet - player.current_bet)
            player.current_bet = self.current_bet
            self.pot += (self.current_bet - player.current_bet)
            player.last_action = action
            player.last_bet_size = self.current_bet - player.current_bet
            self.button = (self.button + 1) % self.num_players
            return self.get_observation(), 0, self.game_over, {}
        elif action == 2:
            if self.current_bet >= player.stack:
                self.players[self.button].stack -= player.stack
                self.pot += player.stack
            else:
                self.players[self.button].stack -= (self.last_raise - player.current_bet)
                self.pot += (self.last_raise - player.current_bet)
                player.last_bet_size = self.last_raise - player.current_bet
                self.current_bet = self.last_raise
                self.last_raise = max(self.last_raise * 2, 100)
            player.current_bet = self.current_bet
            player.last_action = action
            self.last_to_act = player
            self.button = (self.button + 1) % self.num_players
            return self.get_observation(), 0, self.game_over, {}

    def get_observation(self):
        player = self.players[self.button]
        return {
            "hand": (player.hand[0], player.hand[1]),
            "board": tuple(self.board),
            "position": player.position,
            "stack": player.stack,
            "current_bet": self.current_bet,
            "last_action": player.last_action,
            "last_bet_size": player.last_bet_size,
            "total_pot": self.pot,
            "current_round": self.current_round,
        }

    def create_deck(self) -> List[Tuple[int, str]]:
        deck = []
        for rank in RANKS:
            for suit in SUITS:
                deck.append((RANKS.index(rank), suit))
        random.shuffle(deck)
        return deck

    def deal_hands(self):
        for player in self.players:
            player.hand = [self.deck.pop(), self.deck.pop()]

    def play_round(self):
        self.current_bet = 0
        self.last_raise = 100
        self.last_to_act = None
        for i in range(self.button + 1, self.button + self.num_players + 1):
            player = self.players[i % self.num_players]
            if player.stack == 0:
                continue
            if self.last_to_act is not None and player.position == self.last_to_act.position:
                break
            obs = self.get_observation()
            print("New observation:", obs)
            action = player.act(obs, self.last_to_act)
            print("action:", action)
            self.step(action)
            if self.current_bet != self.players[self.button].current_bet:
                self.last_to_act = player
                self.current_bet = self.players[self.button].current_bet
                self.last_raise = max(self.last_raise, 2 * self.current_bet)
        self.current_round += 1

    def play_game(self):
        while not self.game_over:
            if self.current_round == 0:
                self.players[2].stack -= 50
                self.players[3].stack -= 100
                self.pot = 150
            if self.current_round == 1:
                self.board = [self.deck.pop(), self.deck.pop(), self.deck.pop()]
            if self.current_round == 2:
                self.board.append(self.deck.pop())
            if self.current_round == 3:
                self.board.append(self.deck.pop())
            if self.current_round == 4:
                self.find_winner()
            elif len([p for p in self.players if p.stack > 0]) == 1:
                self.game_over = True
                self.find_winner()
            else:
                self.play_round()

    def find_winner(self):
        players_in_hand = [p for p in self.players if p.stack > 0]
        if len(players_in_hand) == 1:
            players_in_hand[0].stack += self.pot
            self.pot = 0
        else:
            best_hands = []
            for player in players_in_hand:
                hand = player.hand + self.board
                best_hands.append((self.evaluate_hand(hand), player))
            best_hand_value = max([h[0] for h in best_hands])
            winning_players = [h[1] for h in best_hands if h[0] == best_hand_value]
            for player in winning_players:
                player.stack += self.pot // len(winning_players)
            self.pot = 0

    def evaluate_hand(self, hand: List[Tuple[int, str]]) -> int:
        values = [h[0] for h in hand]
        suits = [h[1] for h in hand]
        flush = len(set(suits)) == 1
        straight = False
        if len(set(values)) == 5:
            if max(values) - min(values) == 4:
                straight = True
        else:
            if set(values) == {12, 0, 1, 2, 3}:
                straight = True
        if straight and flush:
            return 8, max(values)
        if straight:
            return 4, max(values)
        if flush:
            return 5, max(values)
        value_counts = {}
        for v in values:
            if v not in value_counts:
                value_counts[v] = 0
            value_counts[v] += 1
        pairs = []
        for v, c in value_counts.items():
            if c == 2:
                pairs.append(v)
        if len(pairs) == 2:
            return 3, max(pairs)
        if len(pairs) == 1:
            if value_counts[pairs[0]] == 4:
                return 7, pairs[0]
            return 2, pairs[0]
        if len(pairs) == 0:
            if max(values) == 12:
                return 6, 12
            return 1, max(values)

    def is_terminal(self):
        return self.game_over

    def get_valid_actions(self):
        player = self.players[self.button]
        if player.stack == 0:
            return [0]
        if self.current_bet == player.current_bet:
            return [1, 2]
        if self.current_bet > player.current_bet:
            return [0, 1, 2]

    def get_payoff(self):
        if not self.game_over:
            return None
        winners = self.get_players_in_hand()
        if len(winners) == 1:
            return {winners[0]: self.pot}
        else:
            payoffs = {}
            for player in winners:
                payoffs[player] = self.pot // len(winners)
            return payoffs

    def get_players_in_hand(self):
        players_in_hand = []
        for player in self.players:
            if player.stack > 0:
                players_in_hand.append(player)
        return players_in_hand

    def is_chance_node(self):
        return len(self.board) < 5

    def sample_chance_outcome(self):
        if len(self.board) == 0:
            self.board = self.deck[:3]
            self.deck = self.deck[3:]
        elif len(self.board) < 5:
            self.board.append(self.deck.pop())

    def get_infoset(self):
        player = self.players[self.button]
        return f"{player.position}:{player.hand[0]}{player.hand[1]}:{self.board}:{player.stack}:{self.current_bet}:{player.last_action}:{player.last_bet_size}"

    def get_player_to_act(self):
        return self.button

    def get_betting_round(self):
        if len(self.board) == 0:
            return "preflop"
        elif len(self.board) == 3:
            return "flop"
        elif len(self.board) == 4:
            return "turn"
        elif len(self.board) == 5:
            return "river"

    def next_state(self, action):
        next_game = copy.deepcopy(self)
        next_game.step(action)
        return next_game

    def search(self, state, depth):
        if self.isTerminal(state) or depth == 0:
            return self.heuristic(state)

        info_set = self.infoSet(state)
        actions = self.actions(state)
        if not actions:  # Проверка на пустое множество действий
            return 0

        if self.isPlayer1(state):
            score = float('-inf')
            for a in actions:
                next_state = self.nextState(state, a)
                score = max(score, self.search(next_state, depth - 1))
            return score
        else:
            score = float('inf')
            for a in actions:
                next_state = self.nextState(state, a)
                score = min(score, self.search(next_state, depth - 1))
            return score


class Player:
    def __init__(self, position, starting_stack):
        self.position = position
        self.starting_stack = starting_stack
        self.stack = starting_stack
        self.hand = []
        self.current_bet = 0
        self.last_action = None
        self.last_bet_size = 0

    def act(self, obs, last_to_act):
        if last_to_act is None:
            return 2
        if last_to_act == self:
            return 1
        if obs["current_bet"] > self.stack:
            return 0
        if obs["last_action"] == 2:
            return 1
        if obs["last_bet_size"] >= self.stack:
            return 1
        return 2


class AbstractPokerGame(PluribusGame):
    """
    Класс для представления абстрактной версии игры в покер.
    """

    def __init__(self, num_players=2):
        super().__init__(num_players)

    def get_state(self):
        """
        Возвращает текущее состояние игры в виде кортежа, содержащего информацию
        о текущем раунде, ставках, банке и т.д.
        """
        return (self.current_round, self.pot, self.current_bet)

    def get_legal_actions(self, player):
        """
        Возвращает список допустимых действий для данного игрока в текущей игре.
        """
        return [Action.FOLD, Action.CHECK, Action.BET, Action.CALL]

    def perform_action(self, action, player):
        """
        Выполняет переданное действие для данного игрока в текущей игре.
        """
        if action == Action.FOLD:
            self.fold(player)
        elif action == Action.CHECK:
            self.check(player)
        elif action == Action.BET:
            self.bet(player)
        elif action == Action.CALL:
            self.call(player)
        else:
            raise ValueError("Invalid action")

    def get_payoffs(self):
        """
        Возвращает выигрыш каждого игрока в виде списка.
        """
        return [player.stack - self.players[player]['stack'] for player in range(self.num_players)]


class Subgame:
    """
    Класс для представления подигры.
    """

    def __init__(self, game, player, action):
        self.game = game
        self.player = player
        self.action = action

    def get_state(self):
        """
        Возвращает текущее состояние подигры.
        """
        return self.game.get_state()

    def get_legal_actions(self, player):
        """
        Возвращает список допустимых действий для данного игрока в текущей подигре.
        """
        if player == self.player:
            return [self.action]
        else:
            return self.game.get_legal_actions(player)

    def perform_action(self, action, player):
        """
        Выполняет переданное действие для данного игрока в текущей подигре.
        """
        if player == self.player:
            self.action = action
        else:
            self.game.perform_action(action, player)

    def is_terminal(self):
        """
        Проверяет, является ли данная подигра терминальной.
        """
        return self.game.is_terminal()

    def get_payoffs(self):
        """
        Возвращает выигрыш каждого игрока в данном терминальном состоянии подигры.
        """
        return self.game.get_payoffs()

    def get_current_player(self):
        """
        Возвращает текущего игрока в данном состоянии подигры.
        """
        return self.game.get_current_player()

    def get_next_state(self, player, action):
        """
        Возвращает следующее состояние подигры после выполнения переданного действия для данного игрока.
        """
        if player == self.player:
            return Subgame(self.game, player, action)
        else:
            return Subgame(self.game.get_next_state(player, action), self.player, self.action)

class TreeNode:
    def __init__(self, state: None, player: int, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.reward = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child(self, action: int) -> 'TreeNode':
        return self.children.get(action, None)

    def add_child(self, action: int, child_node: 'TreeNode'):
        self.children[action] = child_node

    def update(self, reward: float):
        self.visits += 1
        self.reward += reward

    def get_best_child(self, C: float) -> 'TreeNode':
        best_child = None
        best_score = -float('inf')
        for child in self.children.values():
            if child.visits == 0:
                return child
            score = child.reward / child.visits + C * np.sqrt(np.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

class MonteCarloTreeSearch:
    def __init__(self, state: None, player: int, neural_net: None):
        self.state = state
        self.player = player
        self.neural_net = neural_net
        self.root = TreeNode(state=state, player=player)

    def select_action(self) -> int:
        for i in range(100):
            node = self.select(self.root)
            child = self.expand(node)
            reward = self.simulate(child)
            self.backpropagate(child, reward)
        return self.get_best_action()

    def select(self, node: TreeNode) -> TreeNode:
        while not node.is_leaf():
            node = self.get_best_child(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        action = self.choose_untried_action(node)
        next_state = node.state.get_next_state(action)
        next_player = next_state.get_next_player()
        child = TreeNode(state=next_state, player=next_player, parent=node, action=action)
        node.children[action] = child
        return child

    def simulate(self, node: TreeNode) -> float:
        state = node.state
        player = node.player
        while not state.is_terminal():
            if state.get_current_player() == player:
                legal_actions = state.get_legal_actions(player)
                probs = self.neural_net.predict(state, legal_actions)
                action = np.random.choice(legal_actions, p=probs)
            else:
                action = random.choice(state.get_legal_actions(state.get_current_player()))
            state = state.get_next_state(action)
        return state.get_payoff(player)

    def backpropagate(self, node: TreeNode, reward: float):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def get_best_child(self, node: TreeNode) -> TreeNode:
        C = 1.4
        best_child = None
        best_score = -float('inf')
        for child in node.children.values():
            if child.visits == 0:
                return child
            score = child.reward / child.visits + C * np.sqrt(np.log(node.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def choose_untried_action(self, node: TreeNode) -> int:
        legal_actions = node.state.get_legal_actions(node.player)
        for action in legal_actions:
            if action not in node.children:
                return action
        return random.choice(legal_actions)

    def get_best_action(self) -> int:
        best_child = None
        best_score = -float('inf')
        for child in self.root.children.values():
            score = child.reward / child.visits
            if score > best_score:
                best_score = score
                best_child = child
        return best_child.action


class MCCFRPokerPlayer:
    def __init__(self, num_players: int, iterations: int, neural_net: None):
        self.num_players = num_players
        self.iterations = iterations
        self.cumulative_regret_sum = {p: defaultdict(float) for p in range(num_players)}
        self.cumulative_strategy_sum = {p: defaultdict(float) for p in range(num_players)}
        self.strategy_profiles = []
        self.neural_net = neural_net
        self.reach_probs = {p: defaultdict(float) for p in range(num_players)}

    def get_strategy(self, state: None, player: int, legal_actions: List[int], regret_sum: defaultdict) -> defaultdict:
        """
        Возвращает стратегию игрока в данном состоянии игры.
        """
        strategy = defaultdict(float)
        total_regret = sum(regret_sum[state][a] for a in legal_actions)
        for a in legal_actions:
            if total_regret > 0:
                strategy[a] = max(0, regret_sum[state][a]) / total_regret
            else:
                strategy[a] = 1 / len(legal_actions)
        return strategy

    def get_average_strategy(self, state: None, player: int) -> defaultdict:
        """
        Возвращает среднюю стратегию для данного игрока в данном состоянии игры,
        учитывая только последние 100 итераций обучения.
        """
        strategy_sum = defaultdict(float)
        for iteration in range(self.iterations - 100, self.iterations):
            regret_sum = self.cumulative_regret_sum[player]
            strategy_sum = add_dicts(strategy_sum, self.get_strategy(state, player, state.get_legal_actions(player), regret_sum))
        return normalize_dict(strategy_sum)

    def get_regret_matching_strategy(self, state: None, player: int, legal_actions: List[int]) -> defaultdict:
        """
        Возвращает стратегию игрока, учитывая вероятности достижения состояний игры.
        """
        strategy_sum = defaultdict(float)
        total_reach_prob = sum(self.reach_probs[player].values())
        for s in self.reach_probs[player]:
            state_reach_prob = self.reach_probs[player][s] / total_reach_prob
            strategy = self.get_strategy(s, player, s.get_legal_actions(player), self.cumulative_regret_sum[player])
            for a in legal_actions:
                strategy_sum[a] += state_reach_prob * strategy[a]
        return normalize_dict(strategy_sum)

    def compute_cfv(self, state: None, player: int) -> float:
        """
        Вычисляет CFV для данного игрока в данном состоянии игры.
        """
        if state.is_terminal():
            return state.get_payoff(player)

        legal_actions = state.get_legal_actions(player)
        num_legal_actions = len(legal_actions)

        # вычисляем среднюю стратегию для всех игроков
        avg_strategy = {}
        for p in range(self.num_players):
            avg_strategy[p] = self.get_average_strategy(state, p)

        # вычисляем жадную стратегию для текущего игрока на основе средней стратегии всех игроков
        greedy_strategy = {}
        for a in legal_actions:
            joint_strategy = {p: avg_strategy[p][p] if p == player else avg_strategy[p][a] for p in range(self.num_players)}
            greedy_strategy[a] = 1
            for p in range(self.num_players):
                if p == player:
                    continue
                greedy_strategy[a] *= joint_strategy[p]

        # вычисляем CFV для каждого действия
        cf_values = defaultdict(float)
        for i, action in enumerate(legal_actions):
            next_state = state.get_next_state(action)
            next_player = next_state.get_next_player()
            cf_value = self.compute_cfv(next_state, player)
            cf_values[action] = cf_value
            if player == 0:
                self.cumulative_strategy_sum[player][state][action] += greedy_strategy[action]
            for a in legal_actions:
                regret = cf_value - cf_values[a]
                if player == 0:
                    self.cumulative_regret_sum[player][state][a] += regret
                else:
                    self.cumulative_regret_sum[player][state][a] += -regret

        # вычисляем и возвращаем смешанное равновесие Нэша
        if player == 0:
            self.strategy_profiles.append(StrategyProfile(greedy_strategy, legal_actions, cf_values))
        return sum(greedy_strategy[a] * cf_values[a] for a in legal_actions)

    def train_neural_net(self):
        """
        Обучает глубокую нейронную сеть на основе текущих стратегических профилей и выплат
        с использованием оптимизации нескольких агентов.
        """
        game = AbstractPokerGame(self.num_players)
        abstract_state = game.get_state()
        abstract_legal_actions = game.get_legal_actions(0)
        X = []
        Y = []

        # собираем обучающие данные из текущих стратегических профилей и выплат
        for strategy_profile in self.strategy_profiles:
            strategy = {p: strategy_profile[p].get_action_prob(abstract_state, p) for p in range(self.num_players)}
            strategy_product = 1
            for p in range(self.num_players):
                strategy_product *= strategy[p][p]
            for p in range(self.num_players):
                joint_strategy = {q: strategy[q][q] if q == p else strategy[p][q] for q in range(self.num_players)}
                joint_strategy_product = 1
                for q in range(self.num_players):
                    joint_strategy_product *= joint_strategy[q]
                if p == 0:
                    state = abstract_state.to_canonical_poker_state(abstract_legal_actions, joint_strategy)
                    X.append(state)
                    Y.append(joint_strategy_product * strategy_profile.get_payoff())
                else:
                    abstract_state = game.get_next_state(abstract_legal_actions)
                    abstract_legal_actions = game.get_legal_actions(p)
                    state = abstract_state.to_canonical_poker_state(abstract_legal_actions, joint_strategy)
                    X.append(state)
                    Y.append(joint_strategy_product * strategy_profile.get_payoff())

        # обучаем глубокую нейронную сеть на основе обучающих данных
        self.neural_net.train(X, Y, self.iterations)

    def run(self):
        """
        Запускает алгоритм MCCFR для обучения стратегии покера.
        """
        game = AbstractPokerGame(self.num_players)
        for i in range(self.iterations):
            state = game.get_state()
            self.reach_probs[0][state] = 1
            self.compute_cfv(state, 0)
            self.train_neural_net()
            if i % 100 == 0:
                print(f"Iteration {i}")


def add_dicts(a: defaultdict, b: defaultdict) -> defaultdict:
    """
    Возвращает сумму двух defaultdict.
    """
    result = defaultdict(float)
    for key in a:
        result[key] += a[key]
    for key in b:
        result[key] += b[key]
    return result


def normalize_dict(a: defaultdict) -> defaultdict:
    """
    Нормализует defaultdict так, чтобы сумма значений была равна 1.
    """
    total = sum(a.values())
    return {key: a[key] / total for key in a}


class Pluribus:
    def __init__(self, num_players: int, iterations: int, neural_net: None):
        self.num_players = num_players
        self.iterations = iterations
        self.players = [MCCFRPokerPlayer(num_players, iterations, neural_net) for _ in range(num_players)]

    def train(self):
        """
        Обучает мультиагентную стратегию покера с помощью алгоритма Pluribus.
        """
        for i in range(self.iterations):
            subgame = Subgame(self.num_players)
            for j in range(self.num_players):
                self.players[j].reach_probs[j][subgame.get_state()] = 1
                self.players[j].compute_cfv(subgame.get_state(), j)
                self.players[j].train_neural_net()
                if i % 100 == 0:
                    print(f"Iteration {i}, player {j}")
            subgame.advance()
