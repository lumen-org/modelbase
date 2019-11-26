import unittest
import unittest.mock as mock
import dill
import logging

from mb_modelbase import Model
from mb_modelbase.client import Client

logging.getLogger().setLevel(logging.ERROR)

class MyTestCase(unittest.TestCase):
    def test_init(self):
        with mock.patch("socketio.Client"):
            client = Client.Client("localhost", 1234)
            client.socket.connect.assert_called_once_with("http://localhost:1234")

    def test_disconnect(self):
        with mock.patch("socketio.Client"):
            client = Client.Client("localhost", 1234)
            client.disconnect()
            client.socket.disconnect.assert_called_once_with()

    def test_send_models_raise_type_error(self):
        with mock.patch("socketio.Client"):
            client = Client.Client("localhost", 1234)
        with self.assertRaises(TypeError):
            client.send_models("hallo peter")
        with self.assertRaises(TypeError):
            client.send_models(123)
        with self.assertRaises(TypeError):
            client.send_models(["hallo", "peter", (12, 32.1)])

    def test_send_models_working(self):
        with mock.patch("socketio.Client"):
            client = Client.Client("localhost", 1234)
        model = Model()
        client.send_models([model, model])
        client.socket.emit.assert_called_once_with("models", client.dump_models([model, model])[0],
                                                   callback=client._callback_function)

    def test_send_models_single_working(self):
        model = Model()
        with mock.patch("socketio.Client"):
            client = Client.Client("localhost", 1234)
            client.send_models(model)
            client.socket.emit.assert_called_once_with("models", client.dump_models([model])[0],
                                                       callback=client._callback_function)

    def test_dump_models_single(self):
        model = Model()
        expected_model_list = [model.name]
        _, result_model_list = Client.Client.dump_models([model])
        self.assertEqual(expected_model_list, result_model_list)

    def test_dump_models_multiple(self):
        model = Model()
        expected_model_list = [model.name, model.name]
        _, result_model_list = Client.Client.dump_models([model, model])
        self.assertEqual(expected_model_list, result_model_list)

    def test_dump_models_dump_single(self):
        model = Model()
        expected_dill_models = [dill.dumps(model, dill.HIGHEST_PROTOCOL)]
        result_dumped_models, _ = Client.Client.dump_models([model])
        self.assertEqual(expected_dill_models, result_dumped_models)

    def test_dump_models_dump_multiple(self):
        model = Model()
        expected_dill_models = [dill.dumps(model, dill.HIGHEST_PROTOCOL), dill.dumps(model, dill.HIGHEST_PROTOCOL)]
        result_dumped_models, _ = Client.Client.dump_models([model, model])
        self.assertEqual(expected_dill_models, result_dumped_models)

    def test_dump_models_raise_type_error(self):
        with self.assertRaises(TypeError):
            Client.Client.dump_models("hallo peter")
        with self.assertRaises(TypeError):
            Client.Client.dump_models(123)
        with self.assertRaises(TypeError):
            Client.Client.dump_models(["hallo", "peter", (12, 32.1)])


if __name__ == '__main__':
    unittest.main()
