# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

An ActivityLogger allows to store log data.
"""

import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ActivityLogger:

    def __init__ (self):
        self._logfile = None
        self._filehandler = None
        pass

    def log(self, json_obj):
        """Logs the contents of json_obj.
        json_obj must have key 'logPath' which determines the file to log to.
        """

        if 'logPath' not in json_obj:
            raise KeyError('missing logPath property in json_obj')
        logfile = json_obj['logPath']

        # todo: respect append flag

        # need to change log file?
        if self._logfile is None or self._logfile != logfile:
            if self._filehandler is not None:
                self._filehandler.close()  # close old file
            self._logfile = logfile  # set new log file
            self._filehandler = open(self._logfile, 'a')  # open new logfile and append

        # log it baby!
        self._filehandler.write(str(json_obj) + ',\n')


