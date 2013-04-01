#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import exceptions
import shlex
import itertools
import ipdb
from operator import mul
import scipy.optimize
import sys


class Game(object):
    METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
               'Anneal', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
               'cmaes', 'support_enumeration', 'pne']
# not Newton-CG there is needed jacobian of function

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
        payoffs = self.array[player][strategy]
        max_payoff = np.max(payoffs)
        # numbers of best responses strategies
        brs = [index for index, br in enumerate(payoffs) if br == max_payoff]
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
        view = self.array[(player + 1) % 2][row_index][:, col_index]
        numbers = []
        last_row = np.ones((1, num_supports + 1))
        last_row[0][-1] = 0
        last_column = np.ones((num_supports, 1)) * -1
        for index, payoff in enumerate(np.nditer(view, order=order[player],
                                                 flags=['refs_ok'])):
            numbers.append(payoff.flat[0])
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
                result.append(mne)
        return result

    def cmaes(self, strfitnessfct, N, verbose=False):
        """
        Function minimization via Covariance Matrix Adaptation Evolution
        Strategy
        """
        # input parameters
        xmean = np.random.rand(N)
        sigma = 0.5
        stopfitness = 1e-10
        stopeval = 1e3 * N ** 2
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
        counteval = 0
        iteration = 0
        arx = np.empty([N, lamda])
        arz = np.empty_like(arx)
        arfitness = np.empty(lamda)
        while counteval < stopeval:
            for k in range(lamda):
                arz[:, k] = np.random.randn(N)
                arx[:, k] = xmean + sigma * (np.dot(np.dot(B, D), arz[:, k]))
                arfitness[k] = strfitnessfct(arx[:, k])
                counteval += 1
            # sort by fitness and compute weighted mean into xmean
            arindex = np.argsort(arfitness)
            arfitness = arfitness[arindex]
            xmean = np.dot(arx[:, arindex[:mu]], weights)
            zmean = np.dot(arz[:, arindex[:mu]], weights)
            ps = np.dot((1 - cs), ps) + np.dot((np.sqrt(cs * (2 - cs) * mueff)), np.dot(B, zmean))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lamda)) / chiN < 1.4 + 2 / (N + 1)
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
            if counteval - eigenval > lamda / (c1 + cmu) / N / 10:
                eigenval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eig(C)
                D = np.diag(np.sqrt(D))
            if verbose:
                sys.stdout.write("Iteration: {iteration:<5}, Best: {best:<70}, v_function: {v_function:<6.2e}, Sigma: {sigma:.2e}\r".format(
                    iteration=iteration, best=map(lambda x: round(x, 3), arx[:, arindex[0]]), v_function=arfitness[0], sigma=sigma))
                if iteration % 20 == 0:
                    sys.stdout.write('\n')
            iteration += 1
            if arfitness[0] <= stopfitness:
                if verbose:
                    sys.stdout.write('\n')
                break
            # Escape flat fitness, maybe restart?
            if np.abs(arfitness[0] - arfitness[np.ceil(0.7 * lamda)]) <= stopfitness:
                sigma = sigma * np.exp(0.2 + cs/damps)
        result = scipy.optimize.Result()
        result['x'] = arx[:, arindex[0]]
        result['fun'] = arfitness[0]
        result['nfev'] = counteval
        if counteval < stopeval:
            result['success'] = True
            result['status'] = 0
            result['message'] = "Optimization terminated successfully."
        else:
            result['success'] = False
            result['status'] = 1
            result['message'] = 'Something went wrong.'
        return result

    def v_function(self, strategy_profile):
        """
        xij(p) = ui(si, p_i)
        yij(p) = xij(p) - ui(p)
        zij(p) = max[yij(p), 0]
        v(p) = sum_{i \in N} sum_{1 <= j <= mi} [zij(p)]^2
        """
        v = 0.0
        acc = 0
        negative_penalty = np.sum(map(lambda x: min(x, 0) ** 2, strategy_profile))
        v += negative_penalty
        for player in range(self.num_players):
            u = self.payoff(strategy_profile, player)
            one_penalty = (1 - np.sum(strategy_profile[acc:acc+self.shape[player]])) ** 2
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(strategy_profile, player, player_pure_strategy=(player, pure_strategy))
                y = x - u
                #ipdb.set_trace()
                z = max(y, 0.0)
                v += z ** 2
            v += one_penalty
        return v

    def payoff(self, strategyProfile, player, player_pure_strategy=None, normalize=False):
        """
        Function to compute payoff of given strategyProfile
        @param strategyProfile list of probability distributions
        @param player_pure_strategy tuple (player, strategy) to replace
        current player strategy with pure strategy
        @return np.array of payoffs for each player
        """
        deepStrategyProfile = []
        result = 0.0
        acc = 0
        for player, i in enumerate(self.shape):
            strategy = np.array(strategyProfile[acc:acc+i])
            if player_pure_strategy and player == player_pure_strategy[0]:
                strategy = np.zeros_like(strategy)
                strategy[player_pure_strategy[1]] = 1.0
            if normalize:
                strategy = self.normalize(strategy)
            deepStrategyProfile.append(strategy)
            acc += i
        it = np.nditer(self.array[player], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            product = 1.0
            for player, strategy in enumerate(it.multi_index):
                product *= deepStrategyProfile[player][strategy]
            result += product * self.array[player][it.multi_index]
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

    def findEquilibria(self, method='cmaes', verbose=False):
        """
        Find all equilibria
        @return set of PNE and MNE
        """
        if method == 'pne':
            return self.getPNE()
        elif self.num_players == 2 and method == 'support_enumeration':
            return self.support_enumeration()
        elif method == 'cmaes':
            result = self.cmaes(self.v_function, self.sum_shape, verbose=verbose)
        elif method in self.METHODS:
            result = scipy.optimize.minimize(self.v_function, np.zeros(self.sum_shape), method=method)
        if verbose:
            print result
        if result.success:
            return result.x
        else:
            return None

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
        if len(brackets) == 4:
            # payoff version
            self.players = tokens[brackets[0] + 1:brackets[1]]
            self.num_players = len(self.players)
            self.shape = tokens[brackets[2] + 1:brackets[3]]
            self.shape = map(int, self.shape)
            payoffs_flat = tokens[brackets[3] + 1:brackets[3] + 1 +
                                  reduce(mul, self.shape) * self.num_players]
            payoffs_flat = map(float, payoffs_flat)
            payoffs = []
            for i in range(0, len(payoffs_flat), self.num_players):
                payoffs.append(payoffs_flat[i:i + self.num_players])
        else:
            # outcome verion
            brackets_pairs = []
            for i in brackets:
                if tokens[i] == "{":
                    brackets_pairs.append([i])
                if tokens[i] == "}":
                    pair = -1
                    while len(brackets_pairs[pair]) != 1:
                        pair -= 1
                    brackets_pairs[pair].append(i)
            self.players = tokens[brackets[0] + 1:brackets[1]]
            self.num_players = len(self.players)
            i = 2
            self.shape = []
            while brackets_pairs[i][1] < brackets_pairs[1][1]:
                self.shape.append(brackets_pairs[i][1] - brackets_pairs[i][0] - 1)
                i += 1
            after_brackets = brackets_pairs[i][1] + 1
            i += 1
            outcomes = []
            outcomes.append([0] * self.num_players)
            for i in range(i, len(brackets_pairs)):
                outcomes.append(map(lambda x: float(x.translate(None, ',')), tokens[brackets_pairs[i][0] + 2:brackets_pairs[i][1]]))
            payoffs = [outcomes[out] for out in map(int, tokens[after_brackets:])]
        self.sum_shape = np.sum(self.shape)
        self.array = []
        for player in range(self.num_players):
            self.array.append(np.ndarray(self.shape, dtype=float, order="F"))
        it = np.nditer(self.array[0], flags=['multi_index', 'refs_ok'])
        index = 0
        while not it.finished:
            for player in range(self.num_players):
                self.array[player][it.multi_index] = payoffs[index][player]
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
        result += " ".join(map(lambda x: "\"" + x + "\"", self.players))
        result += " } { "
        result += " ".join(map(str, self.shape))
        result += " }\n\n"
        for payoff in np.nditer(self.array[0], order="F", flags=['refs_ok']):
            for i in payoff.flat[0]:
                    result += str(i) + " "
        return result

    def coordinateToStrategyProfile(self, t):
        """
        Translate tuple form of strategy profile to long, gambit-like format
        @params t tuple to Translate
        @return list of numbers in long format
        """
        result = [0.0] * self.sum_shape
        accumulator = 0
        for index, i in enumerate(self.shape):
            result[t[index] + accumulator] = 1.0
            accumulator += i
        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-m', '--method', default='cmaes', choices=Game.METHODS)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    with open(args.file) as f:
        game_str = f.read()
    g = Game(game_str)
    result = g.findEquilibria(args.method, args.verbose)
    if result is not None:
        print "NE: ", np.round(np.abs(result), decimals=4)
    else:
        sys.exit("Nash equilibrium was not found.")
