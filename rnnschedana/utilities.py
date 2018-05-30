"""Utility functions"""
from datetime import datetime
import copy

class classproperty:
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, type):
        return self.f(type)

class StopWatch:
    def __init__(self, main):
        self._main = main
        self._watches = None

    def start(self, name=None):
        if not self._watches:
            self._watches = {self._main:{'ts':[datetime.now()], 'delta':[], 'total':None}}
            w = self._watches[self._main]
        if name:
            if name not in self._watches:
                self._watches[name] = {'ts':[], 'delta':[], 'total':None}
            w = self._watches[name]
            w['ts'].append(datetime.now())
        return str(w['ts'][-1])

    def stop(self, name=None):
        w = self._watches[self._main]
        if len(w['ts']) < 2:
            w['ts'].append(datetime.now())

            w['delta'].append(delta_time_str((w['ts'][1]-w['ts'][0]).total_seconds()))
        else:
            w['ts'][1] = datetime.now()
            w['delta'][0] = delta_time_str((w['ts'][1]-w['ts'][0]).total_seconds())
        w['total'] = delta_time_str((w['ts'][-1]-w['ts'][0]).total_seconds())

        if name:
            w = self._watches[name]
            w['ts'].append(datetime.now())
            w['delta'].append(delta_time_str((w['ts'][-1]-w['ts'][-2]).total_seconds()))
            w['total'] = delta_time_str((w['ts'][-1]-w['ts'][0]).total_seconds())
        return w['delta'][-1]

    def __str__(self):
        return str(self.logs)

    @property
    def logs(self):
        logs = copy.deepcopy(self._watches)
        for key in logs.keys():
            logs[key]['ts'] = list(map(str, logs[key]['ts']))
        return logs

def delta_time(s):
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return d,h,m,s

def delta_time_str(s):
    return '{}d {}h {}m {}s'.format(*delta_time(s))

def to_bytes(text):
    """python3 -> str = unicode, bytes=encoded, e.g. utf-8"""
    if isinstance(text, str):
        return text.encode('utf-8')
    else:
        return text

def to_str(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')
    else:
        return text

def truncate(string):
    return string[:30] if len(string) > 30 else string

def flatten(l, ltypes=(list, tuple)):
    """really cool flatten method"""
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop[i]
                i -= 1
                break
            else:
                l[i:i+1] = l[i]
        i += 1

    return l
