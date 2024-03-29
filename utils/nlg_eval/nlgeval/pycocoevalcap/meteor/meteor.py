#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import sys
import os
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    #return s.encode('utf-8')
    return s.encode('GBK')

def dec(s):
    #  return s.decode('utf-8')
    return s.decode('GBK')


class Meteor:

    def __init__(self):
        meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        print(os.path.dirname(os.path.abspath(__file__)))
        self.meteor_p = subprocess.Popen(meteor_cmd,cwd=os.path.dirname(os.path.abspath(__file__)),
                                        env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE
                                        ,shell=True)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                scores.append(float(dec(self.meteor_p.stdout.readline().strip())))
            score = float(dec(self.meteor_p.stdout.readline()).strip())

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line+'\r\n'))
        # self.meteor_p.stdin.write(enc())
        self.meteor_p.stdin.flush() 
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats 
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        with self.lock:
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
