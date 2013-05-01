#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import shlex
import itertools
from operator import mul
import scipy.optimize
import sys
import logging
import cmaes
import ipdb

class Game(object):
    METHODS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
               'Anneal', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
               'cmaes', 'support_enumeration', 'pne', 'cma']
# not Newton-CG there is needed jacobian of function

    def __init__(self, nfg):
        self.read(nfg)
        self.players_zeros = np.zeros(self.num_players)
        self.brs = None
        self.degenerated = None
        self.deleted_strategies = None

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
        self.brs = [set() for i in xrange(self.num_players)]
        for player in xrange(self.num_players):
            p_view = self.shape[:]
            p_view[player] = 1
            # get all possible opponent strategy profiles to 'player'
            for strategy in np.ndindex(*p_view):
                # add to list of best responses
                self.brs[player].update(self.bestResponse(player, strategy))
        # check degeneration of a game
        self.degenerated = self.isDegenerated()
        # PNE is where all player have Best Response
        ne_coordinates = set.intersection(*self.brs)
        result = map(self.coordinateToStrategyProfile, ne_coordinates)
        return result

    def getDominatedStrategies(self):
        """
        @return list of players' dominated strategies
        """
        empty = [slice(None)] * self.num_players
        result = []
        for player in xrange(self.num_players):
            s1 = empty[:]
            strategies = []
            dominated_strategies = []
            for strategy in xrange(self.shape[player]):
                s1[player] = strategy
                strategies.append(self.array[player][s1])
            for strategy in xrange(self.shape[player]):
                dominated = False
                for strategy2 in xrange(self.shape[player]):
                    if strategy == strategy2:
                        continue
                    elif (strategies[strategy] < strategies[strategy2]).all():
                        dominated = True
                        break
                if dominated:
                    dominated_strategies.append(strategy)
            result.append(dominated_strategies)
        return result

    def iteratedEliminationDominatedStrategies(self):
        """
        Eliminates all strict dominated strategies, preserve self.array and
        self.shape in self.init_array and self.init_shape. Stores numbers of
        deleted strategies in self.deleted_strategies.
        """
        self.init_array = self.array[:]
        self.init_shape = self.shape[:]
        self.deleted_strategies = [np.array([], dtype=int) for player in xrange(self.num_players)]
        dominated_strategies = self.getDominatedStrategies()
        while sum(map(len, dominated_strategies)) != 0:
            logging.debug("Dominated strategies to delete: {0}".format(dominated_strategies))
            for player, strategies in enumerate(dominated_strategies):
                for p in xrange(self.num_players):
                    self.array[p] = np.delete(self.array[p], strategies, player)
                for strategy in strategies:
                    original_strategy = strategy
                    while original_strategy in self.deleted_strategies[player]:
                        original_strategy += 1
                    self.deleted_strategies[player] = np.append(self.deleted_strategies[player], original_strategy)#strategy + np.sum(self.deleted_strategies[player] <= strategy))
                self.shape[player] -= len(strategies)
            self.sum_shape = sum(self.shape)
            dominated_strategies = self.getDominatedStrategies()
        for player in xrange(self.num_players):
            self.deleted_strategies[player].sort()

    def isDegenerated(self):
        """
        @return True|False if game is said as degenerated
        """
        if self.num_players != 2:
            return False
        if self.brs is None:
            self.getPNE()
        num_brs = [len(x) for x in self.brs]
        num_strategies = [reduce(mul, self.shape[:k] + self.shape[(k+1):]) for k in xrange(self.num_players)]
        if num_brs != num_strategies:
            return True
        else:
            return False

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

        @params combination combination of strategies to make equation set
        @params player number of player for who the equation matrix will be done
        @params num_supports number of supports for players
        @return equation matrix for solving in e.g. np.linalg.solve
        """
        row_index = np.zeros(self.shape[0], dtype=bool)
        col_index = np.zeros(self.shape[1], dtype=bool)
        row_index[list(combination[0])] = True
        col_index[list(combination[1])] = True
        numbers = self.array[(player + 1) % 2][row_index][:, col_index]
        last_row = np.ones((1, num_supports + 1))
        last_row[0][-1] = 0
        last_column = np.ones((num_supports, 1)) * -1
        if player == 0:
            numbers = numbers.T
        numbers = np.hstack((numbers, last_column))
        numbers = np.vstack((numbers, last_row))
        return numbers

    def support_enumeration(self):
        """
        Computes NE of 2 players nondegenerate games_result

        @return set of NE computed by method support enumeration
        """
        result = self.getPNE()
        # for every numbers of supports
        for num_supports in xrange(2, min(self.shape) + 1):
            logging.debug("Support enumearation for num_supports: {0}".format(num_supports))
            supports = []
            equal = [0] * num_supports
            equal.append(1)
            # all combinations of support length num_supports
            for player in xrange(self.num_players):
                supports.append(itertools.combinations(
                    xrange(self.shape[player]), num_supports))
            # cartesian product of combinations of both player
            for combination in itertools.product(supports[0], supports[1]):
                mne = []
                is_mne = True
                # for both player compute set of equations
                for player in xrange(self.num_players):
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
                    player_strategy_profile = np.zeros(self.shape[player])
                    player_strategy_profile[list(combination[player])] = probabilities
                    # best response condition
                    for pure_strategy in combination[(player + 1) % 2]:
                        if not any(br[(player + 1) % 2] == pure_strategy
                                   and br[player] in combination[player]
                                   for br in self.brs[player]):
                            is_mne = False
                            break
                    mne.extend(player_strategy_profile)
                if is_mne:
                    result.append(mne)
        return result

    def v_function(self, strategy_profile):
        """
        Lyapunov function. If v_function(p) == 0 then p is NE.

        xij(p) = ui(si, p_i)
        yij(p) = xij(p) - ui(p)
        zij(p) = max[yij(p), 0]
        v(p) = sum_{i \in N} sum_{1 <= j <= mi} [zij(p)]^2 + penalty

        @params strategy_profile list of parameters to function
        @return value of v_function in given strategy_profile
        """
        v = 0.0
        acc = 0
        deep_strategy_profile = self.strategy_profile_to_deep(strategy_profile)
        deep_strategy_profile = self.normalize_deep_strategy_profile(deep_strategy_profile)
        #strategy_profile_repaired = np.clip(strategy_profile, 0, 1)
        #out_of_box_penalty = np.sum((strategy_profile - strategy_profile_repaired) ** 2)
        #v += out_of_box_penalty * 10
        for player in range(self.num_players):
            u = self.payoff(deep_strategy_profile, player)
            #one_sum_penalty = (1 - np.sum(strategy_profile[acc:acc+self.shape[player]])) ** 2
            #v += one_sum_penalty
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(deep_strategy_profile, player, pure_strategy)
                z = x - u
                g = max(z, 0.0)
                v += g ** 2
        return v
    
    def payoff(self, strategy_profile, player, pure_strategy=None):
        """
        Function to compute payoff of given strategy_profile

        @param strategy_profile list of probability distributions
        @param pplayer player for who the payoff is computated
        @param pure_strategy if not None pplayer strategy will be replaced
        by pure_strategy
        @param normalize use normalization to strategy_profile
        @return value of payoff
        """
        result = 0.0
        if len(strategy_profile) == self.num_players:
            deep_strategy_profile = strategy_profile[:]
            if pure_strategy is not None:
                new_strategy = np.zeros_like(deep_strategy_profile[player])
                new_strategy[pure_strategy] = 1.0
                deep_strategy_profile[player] = new_strategy
        elif len(strategy_profile) == self.sum_shape:
            deep_strategy_profile = self.strategy_profile_to_deep(strategy_profile)
        else:
            raise Exception("Length of strategy_profile: '{0}', does not match.")
        product = reduce(lambda x, y: np.tensordot(x, y, 0), deep_strategy_profile)
        result = np.sum(product * self.array[player])
        return result

    def strategy_profile_to_deep(self, strategy_profile):
        """
        Convert strategy_profile to deep_strategy_profile.
        It means that instead of list of length sum_shape we have got nested
        list of length num_players and inner lists are of shape[player] length

        @param strategy_profile to convert
        @return deep_strategy_profile
        """
        offset = 0
        deep_strategy_profile = []
        for player, i in enumerate(self.shape):
            strategy = strategy_profile[offset:offset+i]
            deep_strategy_profile.append(strategy)
            offset += i
        return deep_strategy_profile

    def normalize(self, strategy):
        """
        Normalize strategy_profile to asure constraints:
        for all strategies sum p(si) = 1
        p(si) >= 0.0

        @param strategy np.array of probability distribution for one player
        @returns np.array normalized strategy distribution
        """
        return np.abs(strategy) / np.sum(np.abs(strategy))

    def normalize_deep_strategy_profile(self, deep_strategy_profile):
        for index, strategy in enumerate(deep_strategy_profile):
            deep_strategy_profile[index] = self.normalize(strategy)
        return deep_strategy_profile


    def normalize_strategy_profile(self, strategy_profile):
        """
        Normalize whole strategy profile by strategy of each player

        @parameter strategy_profile to be normalized
        @return normalized strategy_profile
        """
        result = []
        acc = 0
        for i in self.shape:
            strategy = np.array(strategy_profile[acc:acc+i])
            result.extend(self.normalize(strategy))
            acc += i
        return result

    def findEquilibria(self, method='cmaes'):
        """
        Find all equilibria, using method

        @params method method from Game.METHODS to be used
        @return list of NE(list of probabilities)
        """
        if method == 'pne':
            result = self.getPNE()
            if len(result) == 0:
                return None
            else:
                return result
        elif self.num_players == 2 and method == 'support_enumeration':
            result = self.support_enumeration()
            if len(result) == 0:
                return None
            else:
                return result
        elif method == 'cmaes':
            result = cmaes.fmin(self.v_function, self.sum_shape)
        elif method in self.METHODS:
            result = scipy.optimize.minimize(self.v_function,
                                             np.random.rand(self.sum_shape),
                                             method=method, tol=1e-10,
                                             options={"maxiter":1e3 * self.sum_shape ** 2})
        logging.info(result)
        #self.degenerated = self.isDegenerated()
        if result.success:
            r = []
            r.append(result.x)
            return r
        else:
            return None

    def read(self, nfg):
        """
        Reads game in .nfg format and stores data to class variables.
        Can read nfg files in outcome and payoff version

        @param nfg string with nfg formated game
        """
        tokens = shlex.split(nfg)
        #preface = ["NFG", "1", "R"]
        #if tokens[:3] != preface:
            #raise exceptions.FormatError(
                #"Input string is not valid nfg format")
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
            for i in xrange(0, len(payoffs_flat), self.num_players):
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
            for i in xrange(i, len(brackets_pairs)):
                outcomes.append(map(lambda x: float(x.translate(None, ',')), tokens[brackets_pairs[i][0] + 2:brackets_pairs[i][1]]))
            payoffs = [outcomes[out] for out in map(int, tokens[after_brackets:])]
        self.sum_shape = np.sum(self.shape)
        self.array = []
        for player in xrange(self.num_players):
            self.array.append(np.ndarray(self.shape, dtype=float, order="F"))
        it = np.nditer(self.array[0], flags=['multi_index', 'refs_ok'])
        index = 0
        while not it.finished:
            for player in xrange(self.num_players):
                self.array[player][it.multi_index] = payoffs[index][player]
            it.iternext()
            index += 1

    def __str__(self):
        """
        Output in nfg payoff format.

        @return game in nfg payoff format
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

        @params t tuple to translate
        @return list of numbers in long format
        """
        result = [0.0] * self.sum_shape
        offset = 0
        for index, i in enumerate(self.shape):
            result[t[index] + offset] = 1.0
            offset += i
        return result

    def printNE(self, nes, payoff=False, checkNE=False):
        """
        Print Nash equilibria with with some statistics

        @params nes list of nash equilibria
        @params payoff print also informations about players payoff
        """
        result = ""
        success = True
        if self.degenerated:
            logging.warning("Game is degenerated")
        for index, ne in enumerate(nes):
            ne = self.normalize_strategy_profile(ne)
            print_ne = list(ne)
            # assure that printed result are in same shape as self.init_shape
            if self.deleted_strategies is not None:
                acc = 0
                for player in xrange(self.num_players):
                    for deleted_strategy in self.deleted_strategies[player]:
                        print_ne.insert(acc + deleted_strategy, 0.0)
                    acc += self.init_shape[player]
            probabilities = map(str, print_ne)
            result += "NE " + ", ".join(probabilities)
            if index != len(nes) - 1:
                result += '\n'
            if payoff:
                s = []
                for player in xrange(self.num_players):
                    s.append("{0}: {1:.3f}".format(self.players[player], self.payoff(ne, player)))
                result += "Payoff " + ", ".join(s) + "\n"
            if checkNE:
                #self.checkNE(map(lambda x: round(x, 4), ne))
                if not self.checkNE(ne):
                    success = False
        return result, success

    def checkNE(self, strategy_profile, num_tests=1000, accuracy=1e-4):
        """
        Function generates random probability distribution for players and
        check if strategy_profile is really NE. If the payoff will be bigger
        it's not the NE.

        @param strategy_profile check if is NE
        @return True if strategy_profile passed test, False otherwise
        """
        payoffs = []
        deep_strategy_profile = []
        acc = 0
        for player, i in enumerate(self.shape):
            strategy = np.array(strategy_profile[acc:acc+i])
            deep_strategy_profile.append(strategy)
            acc += i
            #payoffs
            payoffs.append(self.payoff(strategy_profile, player))
        for player in xrange(self.num_players):
            dsp = deep_strategy_profile[:]
            empty_strategy = [0.0] * self.shape[player]
            for strategy in xrange(self.shape[player]):
                es = empty_strategy[:]
                es[strategy] = 1.0
                dsp[player] = es
                current_payoff = self.payoff(dsp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning('Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(player, dsp[player], payoffs[player],
                                    current_payoff, payoffs[player] - current_payoff))
                    logging.warning("NE test failed")
                    return False
            for i in xrange(num_tests):
                dsp[player] = self.normalize(np.random.rand(self.shape[player]))
                current_payoff = self.payoff(dsp, player)
                if (current_payoff - payoffs[player]) > accuracy:
                    logging.warning('Player {0} has better payoff with {1}, previous payoff {2}, current payoff {3}, difference {4}. '.format(player, dsp[player], payoffs[player],
                                    current_payoff, payoffs[player] - current_payoff))
                    logging.warning("NE test failed")
                    return False
        logging.info("NE test passed")
        return True


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-m', '--method', default='cmaes', choices=Game.METHODS)
    parser.add_argument('-e', '--elimination', action='store_true', default=False)
    parser.add_argument('-p', '--payoff', action='store_true', default=False)
    parser.add_argument('-c', '--checkNE', action='store_true', default=False)
    parser.add_argument('--log', default="CRITICAL", choices=("DEBUG", "INFO",
                                                             "WARNING", "ERROR",
                                                             "CRITICAL"))
    parser.add_argument('--log-file', default=None)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None),
                        format="%(levelname)s, %(asctime)s, %(message)s", filename=args.log_file)

    with open(args.file) as f:
        game_str = f.read()
    start = time.time()
    g = Game(game_str)
    logging.debug("Reading the game took: {0} s".format(time.time() - start))
    if args.elimination:
        g.iteratedEliminationDominatedStrategies()
    result = g.findEquilibria(args.method)
    if result is not None:
        text, success =  g.printNE(result, payoff=args.payoff, checkNE=args.checkNE)
        if success:
            print text
        else:
            sys.exit("Nash equilibrium did not pass the test.")
    else:
        sys.exit("Nash equilibrium was not found.")
