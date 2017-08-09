class ConfigNames(object):
    def __init__(self, names):
        for name in names:
            self.__dict__[name] = name

_CONFIG_FILE = 'ddd.cfg'
CONFIG = {}
with open(_CONFIG_FILE) as f:
    cfg = f.readlines()
for line in cfg:
    name, value = line.split('=')
    CONFIG[name] = value
NAMES = ConfigNames(names=tuple(CONFIG.keys()))