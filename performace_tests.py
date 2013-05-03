#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import subprocess
import time
import pylab
import pickle
import game
import os
import numpy as np
import ipdb
import json

GAMES_DIR = "/home/psebek/projects/bp/neng/games/"
GAMES_DIR_TWOP = "/home/psebek/projects/bp/neng/games/twop/"
GAMES_DIR_PNE = "/home/psebek/projects/bp/neng/games/pne/"

RESULT_DIR_TWOP = "/home/psebek/projects/bp/neng/games/twop/result/"

SCRIPT = "/home/psebek/projects/bp/neng/game.py"

TWOP = range(2,25)
RESULT_TWOP_DIR = "/home/psebek/projects/bp/neng/tests_results/twop/"
RESULT_PNE_DIR =  "/home/psebek/projects/bp/neng/tests_results/pne/"
RESULT_LYAPUNOV_DIR =  "/home/psebek/projects/bp/neng/tests_results/lyapunov/"


METHODS = ['L-BFGS-B', 'SLSQP', 'cmaes']

MAIN_GAMES = ['coord333.nfg', 'coord4.nfg','2x2x2.nfg', '2x2x2x2.nfg', '2x2x2x2x2.nfg', '5x5x5.nfg', '8x8x8.nfg']
MAIN_GAMES_NAMES = ['coord333', 'coord4','2x2x2', '2x2x2x2', '2x2x2x2x2', '5x5x5', '8x8x8']

PLOT_LINES = ['rs--', 'g^--', 'bo--']

TEST_NE = True
CHECK_RETURN_CODE = True

REPEAT = 1

def readNEs(ne_str):
    result = []
    nes = ne_str.split('\n')
    nes = filter(None, nes)
    for ne in nes:
        result.append(map(float, ne[3:].split(',')))
    return result

def array_close_in(array, list_arrays):
    for i in list_arrays:
        if np.allclose(array, i, atol=1e-3):
            return True
    return False

def run_command(command):
    #try:
    start = time.time()
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    #except KeyboardInterrupt:
        #return None, None, None
    if CHECK_RETURN_CODE and p.returncode != 0:
        t = None
        print "Return code:", p.returncode, ", err:", err
    else:
        t = time.time() - start
    return t, out, err

def test_games(games, method, out_file, games_dir, result_dir = None, trim='normalization'):
    result_dir = games_dir + 'result/'
    times = []
    success = []
    global REPEAT
    for game_name in games:
        failed = 0
        rep_t = []
        for i in range(REPEAT):
            print method + ":" + games_dir + game_name
            if method == 'gambit-enummixed':
                t, out, err = run_command([method, "-q" , games_dir + game_name, "-d %d" % 10])
            elif method == 'gambit-enumpure':
                t, out, err = run_command([method, "-q" , games_dir + game_name])
            else:
                if trim == 'normalization':
                    t, out, err = run_command([SCRIPT, "-f=%s" % games_dir + game_name, "-m=%s" % method])
                else:
                    t, out, err = run_command([SCRIPT, "-f=%s" % games_dir + game_name, "-m=%s" % method, '-t=penalization'])
            if t is not None and TEST_NE and out:
                with open(games_dir + game_name) as f:
                    g = game.Game(f.read())
                nes = readNEs(out)
                if result_dir is not None:
                    with open(result_dir + game_name) as f:
                        result = readNEs(f.read())
                    if method not in METHODS and len(nes) != len(result):
                        print "Not enough results"
                        t = None
                for ne in nes:
                    if g.checkNE(ne, num_tests=200, accuracy=1e-3) == False:
                        t = None
                        print "Test not passed: ", ne
                        break
                    if result_dir is not None:
                        if not array_close_in(ne, result):
                            print ne, ", not in:", result
                            t = None
                            break
            if t is None:
                failed += 1
            else:
                rep_t.append(t)
        try:
            t = sum(rep_t) / len(rep_t)
        except ZeroDivisionError:
            t = None
        times.append(t)
        with open(out_file,'w') as f:
            pickle.dump(times, f)
        print "Succeed:", REPEAT - failed, ", Failed:", failed, ", Time :", t
        success.append((REPEAT - failed) / REPEAT)

    return times, success
       
def save_plot(l, xlabel=None, title=None, filename=None, xticks_labels=None, legend_loc='upper left', bottom_adjust=None, yscale='linear', ymin=0.0, ymax_factor=1.1):
    params = {
                'backend': 'ps',
                #'text.usetex': True,'
                'text.latex.unicode': True,
             }
    pylab.rcParams.update(params)
    pylab.figure(1, figsize=(6,4))
    pylab.grid(True)
    pylab.xlabel(xlabel)
    pylab.ylabel(u"čas [s]")
    pylab.title(title)
    pylab.xlim(0, max(l[0][0][0]) * 1.1)
    if xticks_labels is not None:
        pylab.xticks(l[0][0][0], xticks_labels, rotation=25, size='small', horizontalalignment='right')
    ymax = 0.0
    for i in l:
        pylab.plot(*i[0], **i[1])
        ymax = max(ymax,max(i[0][1]))
    pylab.ylim(ymin, ymax * ymax_factor)
    if bottom_adjust is not None:
        pylab.subplots_adjust(bottom=bottom_adjust)
    pylab.yscale(yscale)
    pylab.legend(loc=legend_loc)
    pylab.savefig(filename, format='eps')
    pylab.clf()


def two_players(max_game):
    games_two_range = range(2,max_game)
    games = [str(i) + ".nfg" for i in games_two_range]
    games_two_gambit, success_gambit = test_games(games, "gambit-enummixed", RESULT_TWOP_DIR + "gambit_enummixed.times", GAMES_DIR_TWOP, RESULT_DIR_TWOP)
    games_two_neng, success_neng = test_games(games, "support_enumeration", RESULT_TWOP_DIR + "support_enumeration.times", GAMES_DIR_TWOP, RESULT_DIR_TWOP)
    twop = {
            "games_range": games_two_range,
            "games": games,
            "gambit-enummixed":{
                "time": games_two_gambit,
                "success": success_gambit
                },
            "neng": {
                "time": games_two_neng,
                "success": success_neng
                }
            }
    with open(RESULT_TWOP_DIR+"twop.json", "w") as f:
        json.dump(twop, f, indent=1)
    l = [[[games_two_range, games_two_gambit, 'rs--'], {'label': 'gambit-enummixed'}], [[games_two_range, games_two_neng, 'bo:'],{'label':'neng: support enumeration'}]]
    save_plot(l, xlabel=u'Počet strategií hráčů', title=u'Smíšené Nashovo ekvilibrium metodou vyčíslení domén', filename='plots/sup_enum.eps', yscale='log', ymin=1e-3, ymax_factor=100)

def pne(players):
    global CHECK_RETURN_CODE
    CHECK_RETURN_CODE = False
    result = {}

    games2_range = range(5,151,5)
    games3_range = range(5,101,5)
    games4_range = range(5,36,5)
    games5_range = range(2,23,2)
    games6_range = range(2,13,2)
    games7_range = range(2,11,2)
    games_range = [None, None, games2_range, games3_range, games4_range, games5_range, games6_range, games7_range]
    for players in range(2,players+1):
        games = ["p{0}a{1}.nfg".format(players, i) for i in games_range[players]]
        games_gambit, success_gambit = test_games(games, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p2.times", GAMES_DIR_PNE)
        games_neng, success_neng = test_games(games, "pne", RESULT_PNE_DIR + "p{0}.times".format(players), GAMES_DIR_PNE)
        l = [[[games_range[players], games_gambit, PLOT_LINES[0]], {'label': 'gambit_enumpure'}],[[games_range[players], games_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
        save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro {0} hráče'.format(players), filename='plots/pne{0}.eps'.format(players))
        result['games{0}'.format(players)] = {

                'games_range': games_range[players],
                'games': games,
                'gambit-enumpure':{
                    'time': games_gambit,
                    'success': success_gambit,            
                                },
                'neng':{
                    'time': games_neng,
                    'success': success_neng,            
                        }
                            }
        
    with open(RESULT_PNE_DIR+"pne.json", "w") as f:
        json.dump(result, f, indent=1)
    return 

def lyapunov():
    result = {}
    l = []
    for index, method in enumerate(('cmaes',)):
        print
        print method
        result[method] = {}
        result[method]["time"], result[method]["success"] = test_games(MAIN_GAMES, method, RESULT_LYAPUNOV_DIR+method, GAMES_DIR)
        l.append([[range(1, len(MAIN_GAMES) + 1), result[method]["time"], PLOT_LINES[index]], {'label':method.upper()}])
    result['games'] = MAIN_GAMES
    result['games_range'] = range(1, len(MAIN_GAMES)+1)
    with open(RESULT_LYAPUNOV_DIR + "result.json", "w") as f:
        json.dumps(result, f, indent=1)
    save_plot(l, xlabel=u'Hry', title=u'Smíšené Nashovo ekvilibrium', filename='plots/mne.eps', xticks_labels=MAIN_GAMES_NAMES, bottom_adjust=0.2)


def main():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int )
    parser.add_argument('-p', type=int)
    parser.add_argument('-l', action='store_true')
    parser.add_argument('-r', type=int, default=1)
    args = parser.parse_args()
    REPEAT = args.r
    os.environ['TERM'] = 'dumb'
    if args.t:
        two_players(args.t)
    if args.p:
        pne(args.p)
    if args.l:
        lyapunov()
