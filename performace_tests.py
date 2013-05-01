#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import subprocess
import time
import pylab
#import numpy as np
#import scipy.interpolate as ip
import pickle
import game
import ipdb
import os

GAMES_DIR = "/home/psebek/projects/bp/neng/games/"
GAMES_DIR_TWOP = "/home/psebek/projects/bp/neng/games/twop/"
GAMES_DIR_PNE = "/home/psebek/projects/bp/neng/games/pne/"

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

def readNEs(ne_str):
    result = []
    nes = ne_str.split('\n')
    nes = filter(None, nes)
    for ne in nes:
        result.append(map(float, ne[3:].split(',')))
    return result

def run_command(command):
    try:
        start = time.time()
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    except KeyboardInterrupt:
        return None, None, None
    if CHECK_RETURN_CODE and p.returncode != 0:
        t = None
        print "Return code:", p.returncode, ", err:", err
    else:
        t = time.time() - start
    return t, out, err

def test_games(games, method, out_file, games_dir, repeat=1):
    times = []
    for game_name in games:
        failed = 0
        rep_t = []
        for i in range(repeat):
            print method + ":" + games_dir + game_name
            if method == 'gambit-enummixed':
                t, out, err = run_command([method, "-q" , games_dir + game_name, "-d %d" % 10])
            elif method == 'gambit-enumpure':
                t, out, err = run_command([method, "-q" , games_dir + game_name])
            else:
                t, out, err = run_command([SCRIPT, "-f=%s" % games_dir + game_name, "-m=%s" % method])
            if t is not None and TEST_NE and out:
                with open(games_dir + game_name) as f:
                    g = game.Game(f.read())
                nes = readNEs(out)
                for ne in nes:
                    if g.checkNE(ne, num_tests=200, accuracy=1e-3) == False:
                        t = None
                        print "Test not passed: ", ne
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
        print t
        print "Failed: ", failed, ", succeed: ", repeat - failed
    return times
        
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
    #else:
        #pylab.xticks(l[0][0][0])
    ymax = 0.0
    for i in l:
        pylab.plot(*i[0], **i[1])
        ymax = max(ymax,max(i[0][1]))
    pylab.ylim(ymin, ymax * ymax_factor)
    if bottom_adjust is not None:
        pylab.subplots_adjust(bottom=bottom_adjust)
    pylab.yscale(yscale)
    #pylab.plot(*l)
    pylab.legend(loc=legend_loc)
    #pylab.show()
    pylab.savefig(filename, format='eps')
    pylab.clf()
    #pylab.savefig('demo')


def two_players():
    games_two_range = range(2,13)
    #games = [str(i) + ".nfg" for i in range(2,14)]
    #games_two_gambit = test_games(games, "gambit-enummixed", RESULT_TWOP_DIR + "gambit_enummixed.times", GAMES_DIR_TWOP)
    #games_two_neng = test_games(games, "support_enumeration", RESULT_TWOP_DIR + "support_enumeration.times", GAMES_DIR_TWOP)
    games_two_neng = [0.1986689567565918, 0.15797781944274902, 0.1666429042816162, 0.21634912490844727, 0.27965807914733887, 0.6415009498596191, 1.8710598945617676, 6.391333103179932, 22.76628017425537, 86.18408703804016, 324.1226547895434112]
    games_two_gambit = [0.11462116241455078, 0.02067089080810547, 0.005552053451538086, 0.015307188034057617, 0.018110990524291992, 0.015268087387084961, 0.06959390640258789, 0.5670561790466309, 3.531134843826294, 39.50046110153198, None]
    l = [[[games_two_range, games_two_gambit, 'rs--'], {'label': 'gambit-enummixed'}], [[games_two_range, games_two_neng, 'bo:'],{'label':'neng: support enumeration'}]]
    save_plot(l, xlabel=u'Počet strategií hráčů', title=u'Smíšené Nashovo ekvilibrium metodou vyčíslení domén', filename='plots/sup_enum.eps', yscale='log', ymin=1e-3, ymax_factor=100)

def pne():
    global CHECK_RETURN_CODE
    CHECK_RETURN_CODE = False
    #games2 = ["p2a{0}.nfg".format(i) for i in range(5,151,5)]
    #games2_gambit = test_games(games2, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p2.times", GAMES_DIR_PNE)
    #games2_neng = test_games(games2, "pne", RESULT_PNE_DIR + "p2.times", GAMES_DIR_PNE)
    #l = [[[range(5,151,5), games2_gambit, PLOT_LINES[0]], {'label': 'gambit_enumpure'}],[[range(5,151,5), games2_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    #save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 2 hráče', filename='plots/pne2.eps')

    #games3 = ["p3a{0}.nfg".format(i) for i in range(5,101,5)]
    #games3_gambit = test_games(games3, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p3.times", GAMES_DIR_PNE)
    #games3_neng = test_games(games3, "pne", RESULT_PNE_DIR + "p3.times", GAMES_DIR_PNE)
    #l = [[[range(5,101,5), games3_gambit, PLOT_LINES[0]], {'label': 'gambit-enumpure'}],[[range(5,101,5), games3_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    #save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 3 hráče', filename='plots/pne3.eps')

    #games4_range = range(5,36,5)
    #games4 = ["p4a{0}.nfg".format(i) for i in games4_range]
    #games4_gambit = test_games(games4, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p4.times", GAMES_DIR_PNE)
    #games4_neng = test_games(games4, "pne", RESULT_PNE_DIR + "p4.times", GAMES_DIR_PNE)
    ##games4_neng = [0.18393802642822266, 0.5240747928619385, 1.9133491516113281, 5.456552028656006, 12.942892074584961, 26.56980800628662, 49.13551092147827]
    ##games4_gambit = [0.009245157241821289, 0.0762019157409668, 0.3323228359222412, 1.025184154510498, 2.4503719806671143, 5.070526838302612, 9.467783212661743]
    #l = [[[games4_range, games4_gambit, PLOT_LINES[0]], {'label': 'gambit-enumpure'}],[[games4_range, games4_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    #save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 4 hráče', filename='plots/pne4.eps')

    games5_range = range(2,23,2)
    #games5 = ["p5a{0}.nfg".format(i) for i in games5_range]
    #games5_gambit = test_games(games5, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p5.times", GAMES_DIR_PNE)
    #games5_neng = test_games(games5, "pne", RESULT_PNE_DIR + "p5.times", GAMES_DIR_PNE) 
    games5_neng = [8.875601053237915, 0.3250441551208496, 0.7559590339660645, 1.7992889881134033, 4.7639288902282715, 11.232058048248291, 23.90341806411743, 45.398128032684326, 82.58618402481079, 137.03511500358582, 224.2101080417633]
    games5_gambit = [0.006890773773193359, 0.022851943969726562, 0.06631994247436523, 0.2657020092010498, 0.7735369205474854, 1.8992199897766113, 4.132354021072388, 8.001173973083496, 14.327367067337036, 24.526022911071777, 77.54557204246521]
    l = [[[games5_range, games5_gambit, PLOT_LINES[0]], {'label': 'gambit-enumpure'}],[[games5_range, games5_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 5 hráčů', filename='plots/pne5.eps')
    return

    games6 = ["p6a{0}.nfg".format(i) for i in range(2,17,2)]
    games6_gambit = test_games(games6, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p6.times", GAMES_DIR_PNE)
    games6_neng = test_games(games6, "pne", RESULT_PNE_DIR + "p6.times", GAMES_DIR_PNE)
    l = [[[range(5,16,5), games6_gambit, PLOT_LINES[0]], {'label': 'gambit-enumpure'}],[[range(5,16,5), games6_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 6 hráčů', filename='plots/pne6.eps')

    games7 = ["p7a{0}.nfg".format(i) for i in range(2,11,2)]
    games7_gambit = test_games(games7, "gambit-enumpure", RESULT_PNE_DIR + "gambit_enumpure_p7.times", GAMES_DIR_PNE)
    games7_neng = test_games(games7, "pne", RESULT_PNE_DIR + "p7.times", GAMES_DIR_PNE)
    l = [[[range(5,11,5), games7_gambit, PLOT_LINES[0]], {'label': 'gambit-enumpure'}],[[range(5,11,5), games7_neng, PLOT_LINES[-1]], {'label':'neng-pne'}]]
    save_plot(l, xlabel=u'Počet strategií hráče', title=u'Ryzí Nashovo ekvilibrium pro 7 hráčů', filename='plots/pne7.eps')

def lyapunov():
    result = {}
    l = []
    for index, method in enumerate(METHODS):
        print
        print method
        result[method] = test_games(MAIN_GAMES, method, RESULT_LYAPUNOV_DIR+method, GAMES_DIR, repeat=3)
        l.append([[range(1, len(MAIN_GAMES) + 1), result[method], PLOT_LINES[index]], {'label':method.upper()}])
    save_plot(l, xlabel=u'Hry', title=u'Smíšené Nashovo ekvilibrium', filename='plots/mne.eps', xticks_labels=MAIN_GAMES_NAMES, bottom_adjust=0.2)


def main():
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-p', action='store_true')
    parser.add_argument('-l', action='store_true')
    args = parser.parse_args()
    os.environ['TERM'] = 'dumb'
    if args.t:
        two_players()
    if args.p:
        pne()
    if args.l:
        lyapunov()
