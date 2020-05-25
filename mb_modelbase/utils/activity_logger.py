# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

An ActivityLogger allows to store log data.
"""

import json

class ActivityLogger:

    def __init__(self):
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

        # need to change log file?
        if self._logfile is None or self._logfile != logfile:
            if self._filehandler is not None:
                # close old file
                self._filehandler.close()

            # set new log file
            self._logfile = logfile

            # open new logfile and append or overwrite
            replace = json_obj.get('logAppend', True)
            file_mode = 'tw' if replace else 'ta'
            self._filehandler = open(self._logfile, file_mode, 1)  # 1 means line buffering

        # log it baby!
        self._filehandler.write(json.dumps(json_obj) + '\n')
