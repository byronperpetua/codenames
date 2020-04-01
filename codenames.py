import itertools
import jellyfish
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
import scipy

model = pd.read_pickle('numberbatch.gz')
vocab_rank = pd.read_pickle('vocab_rank.gz')
with open('word_list.txt', 'r') as f:
    target_list = [w.strip() for w in f.readlines()]

class CodenamesBot:
    def __init__(self, model, vocab_rank, target_list,
                 board=None,
                 max_rank=25000,
                 initial_clue_candidates=100,
                 max_legal_jaro_winkler=0.8,
                 min_similarity=0.1,
                 max_similarity=0.5,
                 n_targets_exponent=1.5):
        self.model = model
        self.vocab_rank = vocab_rank
        self.target_list = target_list
        self.max_rank = max_rank
        self.max_legal_jaro_winkler = max_legal_jaro_winkler
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.n_targets_exponent = n_targets_exponent
        self.initial_clue_candidates = initial_clue_candidates
        self.stemmer = SnowballStemmer('english').stem
        if board is None:
            self.create_board()
        else:
            self.board = board
        self.precompute_similarity_matrix()
        self.vocab = self.model[self.vocab_rank <= self.max_rank].index
        self.used_clues = {'red': [], 'blue': []}
        
    def create_board(self):
        colors = (['red']*8 + ['blue']*8 + ['neutral']*7 + ['assassin']
                  + (['red'] if np.random.randint(2) else ['blue']))
        np.random.shuffle(colors)
        self.board = pd.Series(
            index=np.random.choice(self.target_list, size=25, replace=False),
            data=colors)
        
    def is_illegal_clue(self, candidates):
        illegal = np.array([False] * len(candidates))
        for t in self.board.index:
            illegal |= candidates.index.str.contains(t)
            illegal |= candidates.index.str.contains(self.stemmer(t))
            illegal |= [c in t for c in candidates.index]
            illegal |= [self.stemmer(c) in t for c in candidates.index]
            illegal |= [jellyfish.jaro_winkler(t, c)
                            > self.max_legal_jaro_winkler
                        for c in candidates.index]
        return illegal
    
    def is_used_clue(self, color, candidates):
        used = np.array([False] * len(candidates))
        for u in self.used_clues[color]:
            used |= candidates.index.str.contains(u)
            used |= candidates.index.str.contains(self.stemmer(u))
            used |= [c in u for c in candidates.index]
            used |= [self.stemmer(c) in u for c in candidates.index]
        return used
        
    def precompute_similarity_matrix(self):
        self.full_mat = self.model.dot(self.model.loc[self.board.index].T)
        self.full_mat[self.full_mat > self.max_similarity] = (
            (self.full_mat - self.max_similarity) * 0.0001
            + self.max_similarity)
        self.full_mat = self.full_mat - self.min_similarity
        self.full_mat[self.full_mat < 0] = self.full_mat * 0.0001
        self.sim_mat = self.full_mat.loc[self.vocab_rank <= self.max_rank, :]

    def clue(self, color, targets, n_clues=5):
        target_mat = self.sim_mat[targets]
        opp_color = 'red' if color == 'blue' else 'blue'
        neutral_mat = self.sim_mat[self.board.index[self.board == 'neutral']]
        opponent_mat = self.sim_mat[self.board.index[self.board == opp_color]]
        assassin_mat = self.sim_mat[self.board.index[self.board == 'assassin']]
        target_scores = target_mat.min(axis='columns')
        neutral_scores = neutral_mat.max(axis='columns').fillna(0) * 0.5
        opponent_scores = opponent_mat.max(axis='columns').fillna(0)
        assassin_scores = assassin_mat.max(axis='columns').fillna(0) * 2
        penalty_scores = (pd.concat([opponent_scores, assassin_scores,
                                     neutral_scores],
                                    axis='columns')
                            .max(axis='columns'))
        scores = target_scores - penalty_scores
        scores_df = pd.DataFrame({'target': target_scores,
                           'penalty': penalty_scores,
                           'score': scores})
        candidates = scores_df.nlargest(self.initial_clue_candidates, 'score')
        df = (candidates[~self.is_illegal_clue(candidates)
                         & ~self.is_used_clue(color, candidates)]
                .nlargest(n_clues, 'score'))
        df['targets'] = pd.Series([targets for i in range(len(df))],
                                   index=df.index)
        df['adj_score'] = df.score * len(targets)**self.n_targets_exponent
        return df
    
    def clue_board(self, color, max_targets=4):
        df_list = []
        for k in range(max_targets):
            for targets in itertools.combinations(
                    self.board.index[self.board == color], k+1):
                print('.', end='', flush=True)
                df_list.append(self.clue(color, list(targets), n_clues=1))
        clue_df = pd.concat(df_list)
        clue_row = clue_df.nlargest(1, 'adj_score')
        clue = clue_row.index[0]
        clue_num = len(clue_row.targets[0])
        self.used_clues[color].append(clue)
        return clue, clue_num
    
    def guess(self, clue):
        return self.full_mat.loc[clue].sort_values(ascending=False)
    
    def record_guess(self, guess, color):
        if self.board[guess] == color:
            r = 'Correct'
        elif self.board[guess] == 'assassin':
            r = 'Assassin'
        elif self.board[guess] == 'neutral':
            r = 'Neutral'
        else:
            r = 'Opponent'
        self.board = self.board.drop(guess)
        return r
        
while True:
    bot = CodenamesBot(model, vocab_rank, target_list)
    player = 'red' if len(bot.board[bot.board == 'red']) == 9 else 'blue'
    while (len(bot.board[bot.board == 'blue']) > 0
            and len(bot.board[bot.board == 'red']) > 0):
        print('Remaining:\n{}'.format(bot.board.value_counts()))
        print('Board: {}'.format(bot.board.index.values))
        print('Thinking of a clue for {}'.format(player), end='')
        print('\n{}'.format(bot.clue_board(player)))
        while True:
            guess = input('Guess (press Enter to pass): ')
            if not guess:
                break
            r = bot.record_guess(guess, player)
            print(r)
            if r != 'Correct':
                break
        player = 'red' if player == 'blue' else 'blue'
