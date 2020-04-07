import time

import dill
import socketio
import logging
from collections.abc import Iterable
import atexit

from mb_modelbase import Model

logger = logging.getLogger(__name__)

# suppress logging from socketio
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)


class Client:
    def __init__(self, url, port):
        """
        Class to handle socket connections with the modelbase server

        :param url: url to connect with
        :param port: port at which the socket is listening
        """
        self.website = url
        self.port = port
        self.socket = self._init_socket()

        # decouple cleanup from Client for atexit to work
        def cleanup(socket):
            socket.disconnect()

        # close socket at exit of program
        atexit.register(cleanup, socket=self.socket)

    def _init_socket(self):
        socket = socketio.Client()
        socket.connect("http://{}:{}".format(self.website, self.port))
        return socket

    def send_models(self, models):
        """
        Sends a model or list of models to the back-end server specified at init

        :param models: model or list of models
        """
        def send_models(models):
            dumped_models, model_list = Client.dump_models(models)
            try:
                logger.info("Sending models: {}".format(model_list))
                self.socket.emit("models", dumped_models, callback=self._callback_function)
            except Exception as e:
                raise e

        try:
            if not isinstance(models, Iterable):
                send_models([models])
            else:
                send_models(models)
        except Exception as e:
            logger.warning(e)
            raise e

    @staticmethod
    def dump_models(models):
        dill_models = []
        model_list = []
        for model in models:
            if isinstance(model, Model):
                dill_models.append(dill.dumps(model, dill.HIGHEST_PROTOCOL))
                model_list.append(model.name)
            else:
                raise TypeError("Expect Object of type Model. Got {}".format(type(model)))
        return dill_models, model_list


    def disconnect(self):
        self.socket.disconnect()

    @staticmethod
    def _callback_function(confirmation):
        """
        output confirmation send from server

        :param confirmation: response from server
        """
        logger.info(confirmation)


if __name__ == "__main__":
    from mb_modelbase import ModelBase

    url = "localhost"
    port = 52104
    client = Client(url, port)
    modelbase = ModelBase("test_modelbase", load_all=False, model_dir="data", watchdog=False)
    modelbase.load_all_models()
    models = [modelbase.get(modelname) for modelname in
              modelbase.list_models()]
    client.send_models(models)
    time.sleep(1)
    client.disconnect()
