#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import subprocess
import time
import pylab
import numpy as np
import scipy.interpolate as ip
import pickle
import game

GAMES_DIR = "/home/psebek/projects/bp/neng/games/"
GAMES_DIR_TWOP = "/home/psebek/projects/bp/neng/games/twop/"
GAMES_DIR_PNE = "/home/psebek/projects/bp/neng/games/pne/"

SCRIPT = "/home/psebek/projects/bp/neng/game.py"


TWOP = range(2,25)
RESULT_TWOP_DIR = "/home/psebek/projects/bp/neng/tests_results/twop/"
RESULT_PNE_DIR =  "/home/psebek/projects/bp/neng/tests_results/pne/"
RESULT_LYAPUNOV_DIR =  "/home/psebek/projects/bp/neng/tests_results/lyapunov/"


METHODS = ['L-BFGS-B', 'SLSQP', 'cmaes']

MAIN_GAMES = ['2x2x2.nfg', '2x2x2x2.nfg', '2x2x2x2x2.nfg', 'coord333.nfg', 'coord4.nfg', 'matchingpennies.nfg', 'prisoners.nfg']

#GAMES = {game_name :game.Game(open(GAMES_DIR+game_name).read()) for game_name in MAIN_GAMES}

def readNEs(ne_str):
    result = []
    nes = ne_str.split('\n')
    nes = filter(None, nes)
    for ne in nes:
        result.append(map(float, ne[3:].split(',')))
    return result

def run_command(command, check_return_code=True):
    try:
        start = time.time()
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    except KeyboardInterrupt:
        return None, None, None
    if check_return_code and p.returncode != 0:
        t = None
        print "Return code:", p.returncode, ", err:", err
    else:
        t = time.time() - start
    return t, out, err

def test_games(games, method, out_file, games_dir, check_return_code=True):
    times = []
    for game_name in games:
        print method + ":" + games_dir + game_name
        if method.startswith('gambit'):
            t, out, err = run_command([method, "-q" , games_dir + game_name], check_return_code)
        else:
            t, out, err = run_command([SCRIPT, "-f=%s" % games_dir + game_name, "-m=%s" % method], check_return_code)
        if out:
            with open(games_dir + game_name) as f:
                g = game.Game(f.read())
            nes = readNEs(out)
            for ne in nes:
                if g.checkNE(ne, num_tests=200, accuracy=1e-4) == False:
                    t = None
                    print "Test not passed: ", ne
                    break
        times.append(t)
        with open(out_file,'w') as f:
            pickle.dump(times, f)
        print t
        
def save_plot(x, y, xlabel):
    x = np.array(x)
    y = np.array(y)
    #matplotlib.rc('text', usetex=True)
    #pylab.figure(1)
    #ax = pylab.axes([0,10,0, int(max(y))+1])
    inter = ip.pchip(x,y)
    xx = np.linspace(0,8.0,101)
    pylab.ylim(0,2)
    pylab.grid(True)
    pylab.xlabel(xlabel)
    pylab.ylabel("sekund")
    pylab.plot(x, y,'bo', xx, inter(xx), ':')
    pylab.savefig('demo')


def two_players():
    games = [str(i) + ".nfg" for i in range(2,16)]
    test_games(games, "gambit-enumpoly", RESULT_TWOP_DIR + "gambit_enumpoly.times", GAMES_DIR_TWOP)
    test_games(games, "support_enumeration", RESULT_TWOP_DIR + "support_enumeration.times", GAMES_DIR_TWOP)

def pne():
    games2 = ["p2a{0}.nfg".format(i) for i in range(5,51,5)]
    test_games(games2, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p2.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games2, "pne", RESULT_PNE_DIR + "p2.times", GAMES_DIR_PNE, check_return_code=False)

    games3 = ["p3a{0}.nfg".format(i) for i in range(5,51,5)]
    test_games(games3, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p3.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games3, "pne", RESULT_PNE_DIR + "p3.times", GAMES_DIR_PNE, check_return_code=False)

    games4 = ["p4a{0}.nfg".format(i) for i in range(5,41,5)]
    test_games(games4, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p4.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games4, "pne", RESULT_PNE_DIR + "p4.times", GAMES_DIR_PNE, check_return_code=False)

    games5 = ["p5a{0}.nfg".format(i) for i in range(5,31,5)]
    test_games(games5, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p5.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games5, "pne", RESULT_PNE_DIR + "p5.times", GAMES_DIR_PNE, check_return_code=False) 

    games6 = ["p6a{0}.nfg".format(i) for i in range(5,16,5)]
    test_games(games6, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p6.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games6, "pne", RESULT_PNE_DIR + "p6.times", GAMES_DIR_PNE, check_return_code=False)

    games7 = ["p7a{0}.nfg".format(i) for i in (5, 10)]
    test_games(games7, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p7.times", GAMES_DIR_PNE, check_return_code=False)
    test_games(games7, "pne", RESULT_PNE_DIR + "p7.times", GAMES_DIR_PNE, check_return_code=False)

def lyapunov():
    for method in METHODS:
        print
        print method
        test_games(MAIN_GAMES, method, RESULT_LYAPUNOV_DIR+method, GAMES_DIR)


def main():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-p', action='store_true')
    parser.add_argument('-l', action='store_true')
    args = parser.parse_args()
    if args.t:
        two_players()
    if args.p:
        pne()
    if args.l:
        lyapunov()
