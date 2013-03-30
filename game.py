#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import exceptions
import shlex
import itertools
import ipdb
from operator import mul
#import cmaes


class Game(object):

    def __init__(self, nfg=""):
        if nfg != "":
            self.read(nfg)
            self.players_zeros = np.zeros(self.num_players)

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

    def cmaes(self, strfitnessfct, N):
        """
        Function minimization via Covariance Matrix Adaptation Evolution
        Strategy
        """
        # input parameters
        xmean = np.random.rand(N)
        sigma = 0.5
        stopfitness = 1e-20
        stopeval = 1e4 * N ** 2
        # strategy parameter setting: selection
        lamda = int(4 + 3 * np.log(N))
        mu = lamda / 2
        weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, int(mu) + 1)])
        mu = int(mu)
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        # strategy parameter setting: adaptation
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
        # initialize dynamic (internal) strategy parameters and constants
        pc = np.zeros((1, N))
        ps = np.zeros_like(pc)
        B = np.eye(N)
        D = np.eye(N)
        C = np.identity(N)
        eigenval = 0
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        # generation loop
        self.counteval = 0
        arx = np.empty([N, lamda])
        arz = np.empty_like(arx)
        arfitness = np.empty(lamda)
        while self.counteval < stopeval:
            for k in range(lamda):
                arz[:, k] = np.random.randn(N)
                # arx[:, k] = np.random.multivariate_normal(xmean, sigma ** 2 * C)
                arx[:, k] = xmean + sigma * (np.dot(np.dot(B, D), arz[:, k]))
                arfitness[k] = strfitnessfct(arx[:, k])
                self.counteval += 1
            # sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(arfitness)
            arfitness = arfitness[arindex]
            # xold = xmean
            xmean = np.dot(arx[:, arindex[:mu]], weights)
            zmean = np.dot(arz[:, arindex[:mu]], weights)
            ps = np.dot((1 - cs), ps) + np.dot((np.sqrt(cs * (2 - cs) * mueff)), np.dot(B, zmean))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * self.counteval / lamda)) / chiN < 1.4 + 2 / (N + 1)
            pc = np.dot((1 - cc), pc) + np.dot(np.dot(hsig, np.sqrt(cc * (2 - cc) * mueff)), np.dot(np.dot(B, D), zmean))
            # adapt covariance matrix C
            C = np.dot((1 - c1 - cmu), C) \
                + np.dot(c1, ((pc * pc.T)
                + np.dot((1 - hsig) * cc * (2 - cc), C))) \
                + np.dot(cmu,
                         np.dot(np.dot(np.dot(np.dot(B, D), arz[:, arindex[:mu]]),
                                np.diag(weights)), (np.dot(np.dot(B, D), arz[:, arindex[:mu]])).T))
            # adapt step size sigma
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            # diagonalization
            if self.counteval - eigenval > lamda / (c1 + cmu) / N / 10:
                eigenval = self.counteval
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eig(C)
                D = np.diag(np.sqrt(D))
                # invsqrtC = np.dot(np.dot(B, np.diag(D ** -1)), B.T)
            if arfitness[0] <= stopfitness:
                break
        return arx[:, arindex[0]]

    def v_function(self, strategy_profile):
        """
        xij(p) = ui(si, p_i)
        yij(p) = xij(p) - ui(p)
        zij(p) = max[yij(p), 0]
        v(p) = sum_{i \in N} sum_{1 <= j <= mi} [zij(p)]^2
        """
        v = 0.0
        u = self.payoff(strategy_profile)
        acc = 0
        negative_penalty = np.sum(map(lambda x: min(x, 0) ** 2, strategy_profile))
        v += negative_penalty
        for player in range(self.num_players):
            one_penalty = (1 - np.sum(np.abs(strategy_profile[acc:acc+self.shape[player]]))) ** 2
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(strategy_profile,
                                player_pure_strategy=(player, pure_strategy))[player]
                y = x - u[player]
                z = max(y, 0.0)
                v += z ** 2
            v += one_penalty
        return v

    def payoff(self, strategyProfile,
               player_pure_strategy=None, normalize=False):
        """
        Function to compute payoff of given strategyProfile
        @param strategyProfile list of probability distributions
        @param player_pure_strategy tuple (player, strategy) to replace
        current player strategy with pure strategy
        @return np.array of payoffs for each player
        """
        deepStrategyProfile = []
        result = np.zeros_like(self.players_zeros)
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
        return np.abs(strategy) / np.sum(np.abs(strategy))

    def normalize_strategy_profile(self, strategy_profile):
        result = []
        acc = 0
        for i in self.shape:
            strategy = np.array(strategy_profile[acc:acc+i])
            result.extend(self.normalize(strategy))
            acc += i
        return result

    def findEquilibria(self, method='cmaes'):
        """
        Find all equilibria
        @return set of PNE and MNE
        """
        if self.num_players == 2 and method == 'support_enumeration':
            return self.support_enumeration()
        elif method == 'cmaes':
            #return self.normalize_strategy_profile(self.cmaes(self.v_function, np.sum(self.shape)))
            return self.cmaes(self.v_function, np.sum(self.shape))

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
    #result = g.cmaes(g.v_function, np.sum(g.shape))
    result = g.findEquilibria()
    print "NE: ", result
    print "payoff: ", g.payoff(result)
    print "evaluation: ", g.counteval
