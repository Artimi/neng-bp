#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import game
import ipdb
import numpy as np

selten = """
NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
{ "Player 1" "Player 2" } { 3 2 }

1 1 0 2 0 2 1 1 0 3 2 0
        """

twoxtwoxtwo = """
NFG 1 R "2x2x2 Example from McKelvey-McLennan, with 9 Nash equilibria, 2 totally mixed" { "Player 1" "Player 2" "Player 3" }

{ { "1" "2" }
{ "1" "2" }
{ "1" "2" }
}
""

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

matching_pennies = """
NFG 1 R "Matching pennies"
{ "A" "B" } { 2 2 }

1 -1 -1 1 -1 1 1 -1
           """


class TestReadPayoff(unittest.TestCase):

    def setUp(self):
        self.game = game.Game(selten)
        self.array = np.array([[(1.0, 1.0), (1.0, 1.0)],
                               [(0.0, 2.0), (0.0, 3.0)],
                               [(0.0, 2.0), (2.0, 0.0)]], dtype=object)
        self.name = "Selten (IJGT, 75), Figure 2, normal form"
        self.players = ["Player 1", "Player 2"]
        self.shape = [3, 2]
        self.num_players = 2

    def test_array(self):
        self.assertEqual([0.0, 3.0], self.game.array[1, 1])

    def test_name(self):
        self.assertEqual(self.name, self.game.name)

    def test_players(self):
        self.assertEqual(self.players, self.game.players)

    def test_shape(self):
        self.assertEqual(self.shape, self.game.shape)

    def test_num_players(self):
        self.assertEqual(self.num_players, self.game.num_players)


class TestReadOutcome(unittest.TestCase):

    def setUp(self):
        self.game = game.Game(twoxtwoxtwo)
        self.name = "2x2x2 Example from McKelvey-McLennan, with 9 Nash equilibria, 2 totally mixed"
        self.players = ["Player 1", "Player 2", "Player 3"]
        self.shape = [2, 2, 2]
        self.num_players = 3

    def test_array(self):
        self.assertEqual([0.0, 0.0, 0.0], self.game.array[1, 1, 1])

    def test_name(self):
        self.assertEqual(self.name, self.game.name)

    def test_players(self):
        self.assertEqual(self.players, self.game.players)

    def test_shape(self):
        self.assertEqual(self.shape, self.game.shape)

    def test_num_players(self):
        self.assertEqual(self.num_players, self.game.num_players)


class TestStrFunction(unittest.TestCase):

    def setUp(self):
        self.game = game.Game(matching_pennies)
        self.game_str_after = """NFG 1 R "Matching pennies"
{ "A" "B" } { 2 2 }

1.0 -1.0 -1.0 1.0 -1.0 1.0 1.0 -1.0 """

    def test_print(self):
        self.assertEqual(self.game_str_after, str(self.game))

class TestPNE(unittest.TestCase):
    
    def setUp(self):
        self.twoxtwoxtwo = game.Game(twoxtwoxtwo)
        self.twoPNE = [[0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                       [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        self.selten = game.Game(selten)
        self.seltenPNE = [[1.0, 0.0, 0.0, 1.0, 0.0]]

    def test_twoxtwoxtwo(self):
        self.assertEqual(self.twoxtwoxtwo.findEquilibria(method='pne'), self.twoPNE)

    def test_selten(self):
        self.assertEqual(self.selten.findEquilibria(method='pne'), self.seltenPNE)


class TestSupportEnumeration(unittest.TestCase):
     
    def setUp(self):
        self.mp = game.Game(matching_pennies)
        self.mpNEs = [[0.5, 0.5, 0.5, 0.5]]
        self.selten = game.Game(selten)
        self.seltenNEs = [[1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.5, 0.5]]

    def test_mp(self):
        self.assertEqual(self.mp.support_enumeration(), self.mpNEs)

    def test_selten(self):
        self.assertEqual(self.selten.support_enumeration(), self.seltenNEs)

class TestPayoff(unittest.TestCase):

    def setUp(self):
        self.mp = game.Game(matching_pennies)
        self.ppayoffs = [np.array([1.0, -1.0]),
                         np.array([-1.0, 1.0]), 
                         np.array([-1.0, 1.0]),
                         np.array([1.0, -1.0])]

    def test_mp_pure(self):
        self.assertTrue((self.mp.payoff([1.0, 0.0, 1.0, 0.0]) == self.ppayoffs[0]).all())
        #self.assertEqual(self.mp.payoff([0.0, 1.0, 1.0, 0.0]).all(), self.ppayoffs[1].all())
        #self.assertEqual(self.mp.payoff([1.0, 0.0, 0.0, 1.0]).all(), self.ppayoffs[2].all())
        #self.assertEqual(self.mp.payoff([0.0, 1.0, 0.0, 1.0]).all(), self.ppayoffs[3].all())
        
    def test_mp_mixed(self):
        self.assertTrue((self.mp.payoff([0.5, 0.5, 0.5, 0.5]) == np.array([0.0, 0.0])).all())


if __name__ == '__main__':
    unittest.main()
