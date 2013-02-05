#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import exceptions
import shlex
import itertools
from operator import mul


# TOKENIZER = re.compile(r"[ \n]+(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")


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
        brs = [index for index, br in enumerate(payoffs) if br[player] == max_payoff]
        for br in brs:
            s = strategy[:]
            s[player] = br
            result.add(tuple(s))  # made whole strategy profile, not just one strategy
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
        result = set.intersection(*self.brs)
        return result

    def getMNE(self):
        """
        Computes MNE
        """
        if self.num_players == 2:
            self.support_enumeration()
        else:
            raise NotImplementedError()

    def get_equation_set(self, combination, player, num_supports):
        """
        Return set of equations for given player and combination of strategies
        for 2 players games in support_enumeration
        """
        row_index = np.zeros(self.shape[0], dtype=bool)
        col_index = np.zeros(self.shape[1], dtype=bool)
        order = ['F', 'C']
        row_index[list(combination[0])] = True
        col_index[list(combination[1])] = True
        view = self.array[row_index][:, col_index]
        numbers = []
        last = [1] * num_supports
        last.append(0)
        for index, payoff in enumerate(np.nditer(view, order=order[player], flags=['refs_ok'])):
            numbers.append(payoff.flat[0][(player + 1) % 2])
            if index % num_supports == num_supports - 1:
                numbers.append(-1.0)
        numbers.extend(last)
        return np.array(numbers, dtype=float).reshape(num_supports + 1, num_supports + 1)

    def support_enumeration(self):
        """
        Computes NE of 2 players nondegenerate games
        """
        self.getPNE()
        result = []
        for num_supports in range(2, min(self.shape) + 1):  # changed 1 to to 2
            supports = []
            equal = [0] * num_supports
            equal.append(1)
            for player in range(self.num_players):
                supports.append(itertools.combinations(range(self.shape[player]), num_supports))
            for combination in itertools.product(supports[0], supports[1]):
                mne = []
                is_mne = True
                # for profile in itertools.product(combination):
                    # if profile not in self.brs
                for player in range(self.num_players):
                    equations = self.get_equation_set(combination, player, num_supports)
                    try:
                        result_p = np.linalg.solve(equations, equal)
                    except np.linalg.LinAlgError:  # not solvable equations
                        is_mne = False
                        break
                    probabilities = result_p[:-1]
                    if not np.all(probabilities >= 0):  # TODO: must be best response
                        is_mne = False
                        break
                    player_strategy_profile = [0.0] * self.shape[player]
                    for index, i in enumerate(combination[player]):
                        player_strategy_profile[i] = probabilities[index]
                    mne.extend(player_strategy_profile)
                if not is_mne:
                    continue
                result.append(tuple(mne))
                print combination, ":", mne
                print "product: ",
                for i in itertools.product(*combination):
                    print i,
                print
        return result

    def findEquilibria(self):
        """
        Find all equilibria
        @return set of PNE and MNE
        """
        return self.getPNE()

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
        brackets = [i for i, x in enumerate(tokens) if x == "{" or x == "}"]
        if len(brackets) != 4:
            raise exceptions.FormatError(
                "Input string is not valid nfg format")
        self.players_name = tokens[brackets[0] + 1:brackets[1]]
        self.num_players = len(self.players_name)
        self.shape = tokens[brackets[2] + 1:brackets[3]]
        self.shape = map(int, self.shape)
        payoffs_flat = tokens[brackets[3] + 1:brackets[3] + 1 + reduce(mul, self.shape) * self.num_players]
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
            for value in payoff.flat:
                for i in value:
                    result += str(i) + " "
        return result

    def tupleToStrategyProfile(self, t):
        """
        Translate tuple form of strategy profile to long, gambit-like format
        @params t tuple to Translate
        @return list of numbers in long format
        """
        result = [0 for i in range(sum(self.shape))]
        accumulator = 0
        for index, i in enumerate(self.shape):
            result[t[index] + accumulator] = 1
            accumulator += i
        return result


if __name__ == '__main__':
    import argparse
    # import pprint as pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-p', '--pickle-file')
    args = parser.parse_args()
    with open(args.file) as f:
        game_str = f.read()
    g = Game(game_str)
#     game_str = """NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
# { "Player 1" "Player 2" } { 3 2 }

# 1 1 0 2 0 2 1 1 0 3 2 0"""
    g.support_enumeration()
    g.getPNE()
    print "BRS: ", g.brs
    # g.getPNE()
    # result = g.findEquilibria()
    # print result
    # for PNE in result:
        # print "NE," +  ",".join(map(str,g.tupleToStrategyProfile(PNE))
