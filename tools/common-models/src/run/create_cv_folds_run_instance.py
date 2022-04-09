class CreateCVFoldsRunInstance(object):
    def __init__(self, num_folds, stratified_class_names, do_shuffle, random_seed):
        self._num_folds = num_folds
        self._stratified_class_names = stratified_class_names
        self._do_shuffle = do_shuffle
        self._random_seed = random_seed

    def __str__(self):
        return "num_folds: {}, stratified_class_names: {}, do_shuffle: {}, random_seed: {}".format(self._num_folds, self._stratified_class_names, self._do_shuffle, self._random_seed)

    @property
    def num_folds(self):
        return self._num_folds

    @property
    def do_shuffle(self):
        return self._do_shuffle

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def stratified_class_names(self):
        return self._stratified_class_names
