import atexit
from pickle import UnpicklingError

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from mb_modelbase.models_core import models as gm

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelWatcher(PatternMatchingEventHandler):
    """
    Modelbase which watches a folder and reacts to newly created *.mdl files
    """
    # files to react to
    patterns = ["*.mdl"]

    def __init__(self, modelbase):
        super().__init__()
        self.modelbase = modelbase

    def on_created(self, event):
        """
        Loads new models into database and ignores already known models

        :param event:  event_type, is_directory, src_path
                    event_type = modified, created, moved, deleted
                    is_directory = True, False
                    src_path = path/to/observer
        :return:
        """
        if not event.is_directory and event.src_path.rsplit("/", 1)[-1][:-4] not in self.modelbase.list_models():
            try:
                model = gm.Model.load(str(event.src_path))
                logger.info("Loaded model from added File {}".format(event.src_path.rsplit("/", 1)[-1]))
            except TypeError as err:
                logger.warning('file "' + event.src_path.rsplit("/", 1)[-1] +
                               '" matches the naming pattern but does not contain a model instance. '
                               'I ignored that file')
                logger.exception(err)
            except UnpicklingError:
                logger.info("Invalid model[pickle] object")
            except Exception as err:
                logger.exception(err)
            else:
                self.modelbase.add(model)
        else:
            if not event.is_directory:
                logger.info("Ignoring Model. Model with same name already exists".format(event.src_path.rsplit("/", 1)[-1]))
                logger.info(self.modelbase.list_models())


class ModelWatchObserver():
    def __init__(self):
        self.observer = Observer()

    def init_watchdog(self, modelbase, path):
        self.observer.schedule(ModelWatcher(modelbase), path=path)
        self.observer.start()
        # cleans the observer up at the end of the program
        atexit.register(self.observer.stop)
