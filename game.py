#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import exceptions
import shlex
import itertools
#import ipdb
from operator import mul


class Game(object):

    def __init__(self, nfg=""):
        if nfg != "":
            self.read(nfg)

    def bestResponse(self, player, strategy):
        """
        Computes bestResponse strategy profile for given opponent strategy and
        player
        @params strategy opponent strategy
        @param player who should respond
        @return set of strategies
        """
        strategy = list(strategy)
        result = set()
        strategy[player] = slice(None)  # all possible strategies for 'player'
        payoffs = self.array[strategy].tolist()
        max_payoff = max(payoffs, key=lambda x: x[player])[player]
        # numbers of best responses strategies
        brs = [index for index, br in enumerate(payoffs)
               if br[player] == max_payoff]
        for br in brs:
            s = strategy[:]
            s[player] = br
            # made whole strategy profile, not just one strategy
            result.add(tuple(s))
        return result

    def getPNE(self):
        """
        Function computes PNE
        @return set of strategy profiles that was computed as pure nash
        equlibria
        """
        # view = [slice(None) for i in range(self.num_players)]
        self.brs = [set() for i in range(self.num_players)]
        for player in range(self.num_players):
            p_view = self.shape[:]
            p_view[player] = 1
            # get all possible opponent strategy profiles to 'player'
            for strategy in np.ndindex(*p_view):
                # add to list of best responses
                self.brs[player].update(self.bestResponse(player, strategy))
        # PNE is where all player have Best Response
        ne_coordinates = set.intersection(*self.brs)
        result = map(self.coordinateToStrategyProfile, ne_coordinates)
        return result

    def get_equation_set(self, combination, player, num_supports):
        """
        Return set of equations for given player and combination of strategies
        for 2 players games in support_enumeration

        This function returns matrix to compute (Nisan algorithm 3.4)
        (I = combination[player])
        \sum_{i \in I} x_i b_{ij} = v
        \sum_{i \in I} x_i = 1
        In matrix form (k = num_supports):
        / b_11 b_12 ... b_1k -1 \ / x_1 \    / 0 \
        | b_21 b_22 ... b_2k -1 | | x_2 |    | 0 |
        | ...  ...  ... ... ... | | ... | =  |...|
        | b_k1 b_k2 ... b_kk -1 | | x_k |    | 0 |
        \ 1    1    ... 1     0 / \ v   /    \ 1 /
        """
        row_index = np.zeros(self.shape[0], dtype=bool)
        col_index = np.zeros(self.shape[1], dtype=bool)
        order = ['F', 'C']
        row_index[list(combination[0])] = True
        col_index[list(combination[1])] = True
        view = self.array[row_index][:, col_index]
        numbers = []
        last_row = np.ones((1, num_supports + 1))
        last_row[0][-1] = 0
        last_column = np.ones((num_supports, 1)) * -1
        for index, payoff in enumerate(np.nditer(view, order=order[player],
                                                 flags=['refs_ok'])):
            numbers.append(payoff.flat[0][(player + 1) % 2])
        numbers = np.array(numbers, dtype=float).reshape(num_supports,
                                                         num_supports)
        numbers = np.hstack((numbers, last_column))
        numbers = np.vstack((numbers, last_row))
        return numbers

    def support_enumeration(self):
        """
        Computes NE of 2 players nondegenerate games
        """
        result = self.getPNE()
        # for every numbers of supports
        for num_supports in range(2, min(self.shape) + 1):
            supports = []
            equal = [0] * num_supports
            equal.append(1)
            # all combinations of support length num_supports
            for player in range(self.num_players):
                supports.append(itertools.combinations(
                    range(self.shape[player]), num_supports))
            # cartesian product of combinations of both player
            for combination in itertools.product(supports[0], supports[1]):
                mne = []
                is_mne = True
                # for both player compute set of equations
                for player in range(self.num_players):
                    equations = self.get_equation_set(combination, player,
                                                      num_supports)
                    try:
                        equations_result = np.linalg.solve(equations, equal)
                    except np.linalg.LinAlgError:  # unsolvable equations
                        is_mne = False
                        break
                    probabilities = equations_result[:-1]
                    # all probabilities are nonnegative
                    if not np.all(probabilities >= 0):
                        is_mne = False
                        break
                    player_strategy_profile = [0.0] * self.shape[player]
                    for index, i in enumerate(combination[player]):
                        player_strategy_profile[i] = probabilities[index]
                    # best response condition
                    for pure_strategy in combination[(player + 1) % 2]:
                        if not any(br[(player + 1) % 2] == pure_strategy
                                   and br[player] in combination[player]
                                   for br in self.brs[player]):
                            is_mne = False
                            break
                    mne.extend(player_strategy_profile)
                if not is_mne:
                    continue
                result.append(tuple(mne))
        return result

    def v_function(self, strategy_profile):
        """
        xij(p) = ui(si, p_i)
        yij(p) = xij(p) - ui(p)
        zij(p) = max[yij(p), 0]
        v(p) = sum_{i \in N} sum_{1 <= j <= mi} [zij(p)]^2
        """
        v = 0.0
        for player in range(self.num_players):
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(strategy_profile,
                                player_pure_strategy=(player, pure_strategy))[player]
                y = x - self.payoff(strategy_profile)[player]
                z = max(y, 0.0)
                v += z ** 2
        return v

    def payoff(self, strategyProfile,
               player_pure_strategy=None, normalize=True):
        """
        Function to compute payoff of given strategyProfile
        @param strategyProfile list of probability distributions
        @param player_pure_strategy tuple (player, strategy) to replace
        current player strategy with pure strategy
        @return np.array of payoffs for each player
        """
        deepStrategyProfile = []
        result = np.zeros(self.num_players)
        acc = 0
        for player, i in enumerate(self.shape):
            strategy = np.array(strategyProfile[acc:acc+i])
            if player_pure_strategy and player == player_pure_strategy[0]:
                strategy = np.zeros_like(strategy)
                strategy[player_pure_strategy[1]] = 1.0
            if normalize:
                strategy = self.normalize(strategy)
            #if np.sum(strategy) != 1.0: # here should be some tolerance
                #raise ValueError("every mixed strategy has to sum to one")
            deepStrategyProfile.append(strategy)
            acc += i
        it = np.nditer(self.array, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            product = 1.0
            for player, strategy in enumerate(it.multi_index):
                product *= deepStrategyProfile[player][strategy]
            result += product * np.array(self.array[it.multi_index])
            it.iternext()
        return result

    def normalize(self, strategy):
        """
        Normalize strategy_profile to asure constraints:
        for all strategies sum p(si) = 1
        p(si) >= 0.0
        """
        return np.abs(strategy) / np.sum(strategy)

    def findEquilibria(self):
        """
        Find all equilibria
        @return set of PNE and MNE
        """
        if self.num_players == 2:
            return self.support_enumeration()
        else:
            raise NotImplementedError()

    def read(self, nfg):
        """
        Reads game in .nfg format and stores data to class variables
        @param nfg string with nfg formated game
        """
        tokens = shlex.split(nfg)
        preface = ["NFG", "1", "R"]
        if tokens[:3] != preface:
            raise exceptions.FormatError(
                "Input string is not valid nfg format")
        self.name = tokens[3]
        brackets = [i for i, x in enumerate(tokens) if x == "{" or x == "}"]
        if len(brackets) != 4:
            raise exceptions.FormatError(
                "Input string is not valid nfg format")
        self.players_name = tokens[brackets[0] + 1:brackets[1]]
        self.num_players = len(self.players_name)
        self.shape = tokens[brackets[2] + 1:brackets[3]]
        self.shape = map(int, self.shape)
        payoffs_flat = tokens[brackets[3] + 1:brackets[3] + 1 +
                              reduce(mul, self.shape) * self.num_players]
        payoffs_flat = map(float, payoffs_flat)
        complete_shape = self.shape[:]
        complete_shape.append(self.num_players)
        payoffs = []
        for i in range(0, len(payoffs_flat), self.num_players):
            payoffs.append(tuple(payoffs_flat[i:i + self.num_players]))
        self.array = np.ndarray(self.shape, dtype=tuple, order="F")
        it = np.nditer(self.array, flags=['multi_index', 'refs_ok'])
        index = 0
        while not it.finished:
            self.array[it.multi_index] = payoffs[index]
            it.iternext()
            index += 1

    def __str__(self):
        """
        Output in nfg format
        @return game in nfg format
        """
        result = "NFG 1 R "
        result += "\"" + self.name + "\"\n"
        result += "{ "
        result += " ".join(map(lambda x: "\"" + x + "\"", self.players_name))
        result += " } { "
        result += " ".join(map(str, self.shape))
        result += " }\n\n"
        for payoff in np.nditer(self.array, order="F", flags=['refs_ok']):
            for i in payoff.flat[0]:
                    result += str(i) + " "
        return result

    def coordinateToStrategyProfile(self, t):
        """
        Translate tuple form of strategy profile to long, gambit-like format
        @params t tuple to Translate
        @return list of numbers in long format
        """
        result = [0] * sum(self.shape)
        accumulator = 0
        for index, i in enumerate(self.shape):
            result[t[index] + accumulator] = 1
            accumulator += i
        return tuple(result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    args = parser.parse_args()
    with open(args.file) as f:
        game_str = f.read()
    g = Game(game_str)
    #g.payoff([0.5, 0.5, 0, 0.5, 0.5], player_pure_strategy=(0,1))
    print g.payoff([1.0, 0.0, 0.0, 0,5, 0.5])
    #print g.v_function([1.0, 0.0, 0.0, 0.5, 0.5])
    #l = g.findEquilibria()
    #print l
    #for i in l:
        #s = ",".join(map(str, i))
        #print "NE," + s
    #print g.normalize([4.0,3.0,2.0])
