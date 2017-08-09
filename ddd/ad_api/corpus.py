import os
from pympi import Elan
import logging

logger = logging.getLogger(__name__)


class Corpus(list):
    def __init__(self, documents=()):
        super().__init__()
        for doc in documents:
            self.append(doc)
        self.add_document = self.append
        self.add_documents = self.extend
        self.remove_document = self.remove

    @staticmethod
    def from_directory(path, **kwargs):
        document_paths = (os.path.join(path, fname) for fname in os.listdir(path))
        documents = (Document(dp, **kwargs) for dp in filter(lambda fn: fn.endswith('eaf'), document_paths))
        return Corpus(documents=documents)


class Document(object):
    def __init__(self, path, min_level='text', ignore_tiers=('character',), lang_name='lang'):
        self._path = path
        self._id = os.path.basename(path)
        self._tok = min_level
        eaf = Elan.Eaf()
        Elan.parse_eaf(file_path=path, eaf_obj=eaf)
        self._name = None  # TODO look-up of names
        ix_times = {i: t for i, t in enumerate(sorted(set(eaf.timeslots.values())))}
        times_ix = {t: i for i, t in ix_times.items()}
        annos = {ix: {} for ix in ix_times}
        tiers = eaf.tiers
        langs = {}
        anno_summary = {}
        for tiername, annotations in tiers.items():
            if tiername not in ignore_tiers:
                anno_summary[tiername] = set()
                for _, avalue in annotations[0].items():
                    anno_summary[tiername].add(avalue[2])
                    anno = Annotation(tiername, avalue[2])
                    start = times_ix[eaf.timeslots[avalue[0]]]
                    end = times_ix[eaf.timeslots[avalue[0]]]
                    for t in range(start, end + 1):
                        annos[t][tiername] = anno
                    if tiername == lang_name:
                        if avalue[2] not in langs:
                            langs[avalue[2]] = 0
                        langs[avalue[2]] += 1
        # filter
        for ix in ix_times:
            if not min_level in annos[ix]:
                o = annos.pop(ix)
                if len(o) > 1:
                    logger.info(('{} '*len(o)).format(*list(o)), '@', ix)
        # normalize indexes
        self._annotations = {nix: annos[ix] for nix, ix in enumerate(annos)}

        self._anno_summary = anno_summary
        self._times = ix_times
        self._langs = langs
        eaf = None

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @property
    def path(self):
        return self._path

    @property
    def token_level(self):
        return self._tok

    @property
    def times(self):
        return self._times

    @property
    def languages(self):
        return self._langs

    @property
    def summary(self):
        return self._anno_summary

    def __getitem__(self, item):
        if item in self._annotations:
            return self._annotations[item]

    def __iter__(self):
        return (v for _, v in self._annotations.items())

    def __repr__(self):
        return self._annotations.__repr__()

    def __str__(self):
        return self._annotations.__str__()
    

class Annotation(object):
    def __init__(self, key, value):
        self._key = key
        self._value = value

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<Annotation: {}={}>'.format(self.key, self.value)
