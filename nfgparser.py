#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import pyparsing as pprs
from pyparsing import Suppress, QuotedString, Combine, Optional, oneOf, Word, nums, OneOrMore, delimitedList, ParseFatalException, Group, stringEnd
from operator import mul
import numpy as np


def validateGame(tokens):
    num_players = len(tokens.players_name)
    mul_shape = reduce(mul,tokens.shape)
    if num_players != len(tokens.shape):
        raise ParseFatalException("Number of players does not match length of shape.")
    if mul_shape * num_players != sum(map(len(tokens.payoffs))):
        raise ParseFatalException("Number of payoffs does not match.")
        
def orderPayoffOutcome(tokens):
    result = []
    outcomes = [[0] * len(tokens.payoff_outcomes[0])] + tokens.payoff_outcomes[:] 
    for i in tokens.payoff_order:
        result.append(outcomes[i])
    return result

def payoffsToNpArrays(tokens):
    num_players = len(tokens.players_name)
    mul_shape = reduce(mul,tokens.shape)
    if len(tokens.payoffs) == mul_shape * num_players:
        payoffs = [tokens.payoffs[i:i+num_players] for i in xrange(0, len(tokens.payoffs), num_players)]
        tokens.payoffs = payoffs
    result = []
    for player in xrange(num_players):
        result.append(np.ndarray(tokens.shape, dtype=float, order="F"))
    it = np.nditer(result[0], flags=['multi_index', 'refs_ok'])
    index = 0
    while not it.finished:
        for player in xrange(num_players):
            result[player][it.multi_index] = tokens.payoffs[index][player]
        it.iternext()
        index += 1
    tokens.payoffs = result
    return tokens

lbrack, rbrack = map(Suppress, "{}")

string = QuotedString('"', multiline=True)
real = Combine(Optional(oneOf("+ -")) + Word(nums) + "." +
               Optional(Word(nums))).setName("real")
real.setParseAction(lambda t: float(t[0]))
integer =  Combine(Optional(oneOf("+ -")) + Word(nums)).setName("integer")
integer.setParseAction(lambda t: int(t[0]))

header = Suppress("NFG 1 R")

game_name = string

players_name = lbrack + OneOrMore(string) + rbrack

shape_list = lbrack + OneOrMore(integer) + rbrack

shape_names =  lbrack + OneOrMore(string) +  rbrack
shape_names.setParseAction(lambda t: len(t))
shape_names_list = lbrack + OneOrMore(shape_names) + rbrack

shape = Group(shape_list | shape_names_list)

payoff_single = real | integer
payoff_list = OneOrMore(payoff_single)

payoff_outcomes = OneOrMore(Group(lbrack + Suppress(QuotedString('"')) + delimitedList(payoff_single) + rbrack))

payoff_order = OneOrMore(integer)
payoff_outcome_list = lbrack + payoff_outcomes("payoff_outcomes") + rbrack + payoff_order("payoff_order")
payoff_outcome_list.setParseAction(orderPayoffOutcome)

payoffs = Group(payoff_list | payoff_outcome_list)

nfgparser = header + game_name('name') + players_name('players_name') + shape('shape') + payoffs('payoffs')  + stringEnd
nfgparser.setParseAction(payoffsToNpArrays)


if __name__ == '__main__':
    s = """
    NFG 1 R "Matching pennies"
    { "A" "B" } { 2 2 }

    1 -1 -1 1 -1 1 1 -1.1
    """

    s2 = """
    NFG 1 R "2x2x2 Example from McKelvey-McLennan, with 9 Nash equilibria, 2 totally mixed" { "Player 1" "Player 2" "Player 3" }

    { { "1" "2" }
    { "1" "2" }
    { "1" "2" }
    }

    {
    { "" 9, 8, 12 }
    { "" 0, 0, 0 }
    { "" 0, 0, 0 }
    { "" 9, 8, 2 }
    { "" 0, 0, 0 }
    { "" 3, 4, 6 }
    { "" 3, 4, 6 }
    { "" 0, 0, 0 }
    }
    1 2 3 4 5 6 7 8

    """
    r = nfgparser.parseString(s)
    print r.payoffs
