import atexit
from pickle import UnpicklingError

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from mb_modelbase.models_core import models as gm

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelWatcher(PatternMatchingEventHandler):
    """Modelbase which watches a folder and reacts to changes.

    Depending on its configuration it reacts on different situations, see below.

    Arguments:
        modelbase: Modelbase
            the modelbase to enable the watcher for.
        reload_on_overwrite: bool, defaults to true.
            Iff true automatically reload a model that was overwritten on disk.
        reolaod_on_creation: bool, defaults to true,
            Iff true automatically load a new model.
        discard_on_delete: bool, defaults to true,
            Iff true automatically delete a model that was deleted from disk also from the
            modelbase.
    """
    # files to react to
    patterns = ["*.mdl"]

    @staticmethod
    def _get_model_name (path):
        return path.rsplit("/", 1)[-1][:-4]

    def __init__(self, modelbase, reload_on_overwrite=True, reload_on_creation=True,
                 discard_on_delete=True):
        super().__init__()
        self.modelbase = modelbase

        self._reload_on_overwrite = reload_on_overwrite
        self._reload_on_creation = reload_on_creation
        self._discard_on_delete = discard_on_delete

    def _load_candidate_model(self, model_path, file_name):
        try:
            model = gm.Model.load(str(model_path))
            logger.info("Loaded model from file {}".format(file_name))
        except TypeError as err:
            logger.warning('file "{}" matches the naming pattern but does not contain a'
                           'model instance. I ignored that file'.format(file_name))
            logger.exception(err)
        except UnpicklingError:
            logger.info("Invalid model[pickle] object")
        except Exception as err:
            logger.exception(err)
        else:
            self.modelbase.add(model)

    def on_modified(self, event):
        if not self._reload_on_overwrite:
            return

        model_path = event.src_path
        file_name = model_path.rsplit("/", 1)[-1]

        self._load_candidate_model(model_path, file_name)

    def on_deleted(self, event):
        if not self._discard_on_delete:
            return
        model_path = event.src_path
        filename = model_path.rsplit("/", 1)[-1]
        model = self.modelbase.get(filename=filename)
        if model is not None:
            self.modelbase.drop(model.name)

    def on_created(self, event):
        """
        Loads new models into database and ignores already known models

        :param event:  event_type, is_directory, src_path
                    event_type = modified, created, moved, deleted
                    is_directory = True, False
                    src_path = path/to/observer
        :return:
        """
        if event.is_directory or not self._reload_on_creation:
            return

        model_path = event.src_path
        model_name = ModelWatcher._get_model_name(model_path)
        file_name = model_path.rsplit("/", 1)[-1]

        if model_name in self.modelbase.list_models() and not self._reload_on_overwrite:
            logger.info("Ignoring Model. Model with same name already exists{}".format(model_name))
            logger.info(self.modelbase.list_models())
            return

        self._load_candidate_model(model_path, file_name)


class ModelWatchObserver:
    def __init__(self):
        self.observer = Observer()

    def init_watchdog(self, modelbase, path, **kwargs):
        self.observer.schedule(ModelWatcher(modelbase, **kwargs), path=path)
        self.observer.start()
        # cleans the observer up at the end of the program
        atexit.register(self.observer.stop)
