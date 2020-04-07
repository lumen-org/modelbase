import pathlib
import unittest
import tempfile

import dill
from mb_modelbase import ModelBase, Model

import logging

logging.disable(logging.WARNING)


class ModelbaseSetupTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        has_exception = False
        try:
            ModelBase(name="my mb", model_dir=self.tmp_dir.name)
        except:
            has_exception = True
        self.assertFalse(has_exception, "Modelbase init failed")

    def test_init_false_dir(self):
        with self.assertRaises(OSError):
            ModelBase(name="my mb", model_dir='test_models')

    def test_watchdog(self):
        with self.subTest("watchdog is online"):
            mb = ModelBase(name="my mb", model_dir=self.tmp_dir.name)
            self.assertTrue(mb.model_watch_observer is not None)
        with self.subTest("watchdog is offline"):
            mb = ModelBase(name="my mb", model_dir=self.tmp_dir.name, watchdog=False)
            self.assertTrue(mb.model_watch_observer is None)

    def test_load_all(self):
        new_model = Model()
        path = pathlib.Path(self.tmp_dir.name).joinpath("first_model.mdl")
        with open(path, 'wb') as output:
            dill.dump(new_model, output, dill.HIGHEST_PROTOCOL)

        with self.subTest("Load all"):
            mb = ModelBase(name="my mb", model_dir=self.tmp_dir.name, watchdog=False, load_all=True)
            self.assertTrue(len(mb.models) == 1)
        with self.subTest("Load nothing"):
            mb = ModelBase(name="my mb", model_dir=self.tmp_dir.name, watchdog=False, load_all=False)
            self.assertTrue(len(mb.models) == 0)


class ModelbaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.first_model = Model("first_model")
        self.second_model = Model("second_model")
        path = pathlib.Path(self.tmp_dir.name).joinpath("first_model.mdl")
        with open(path, 'wb') as output:
            dill.dump(self.first_model, output, dill.HIGHEST_PROTOCOL)
        self.mb = ModelBase(name="my mb", model_dir=self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_all_models(self):
        path = pathlib.Path(self.tmp_dir.name).joinpath("second_model.mdl")
        with open(path, 'wb') as output:
            dill.dump(self.second_model, output, dill.HIGHEST_PROTOCOL)
        models = self.mb.load_all_models(directory=self.tmp_dir.name)

        with self.subTest("Returned all models"):
            self.assertTrue(len(models) == 2)
        with self.subTest("Override existing model"):
            self.assertTrue(len(self.mb.models) == 2)

        expected_dict = {"first_model": self.first_model, "second_model": self.second_model}
        with self.subTest("Different models"):
            self.assertTrue(len({k for k in expected_dict if
                                 k in self.mb.models and expected_dict[k].name == self.mb.models[k].name}) == 2)

    def test_load_all_models_false_dir(self):
        with self.assertRaises(OSError):
            self.mb.load_all_models(directory="test_models")

    def test_load_all_models_false_ext(self):
        path = pathlib.Path(self.tmp_dir.name).joinpath("second_model.mxt")
        with open(path, 'wb') as output:
            dill.dump(self.second_model, output, dill.HIGHEST_PROTOCOL)

        models = self.mb.load_all_models(directory=self.tmp_dir.name)

        with self.subTest("ignored in return models"):
            self.assertTrue(len(models) == 1)

        with self.subTest("ignored in modelbase models"):
            self.assertTrue(len(self.mb.models) == 1)

    @unittest.skip("not build yet")
    def test_save_all_models(self):
        pass

    def test_add(self):
        self.mb.add(self.second_model)
        with self.subTest("added model"):
            self.assertTrue(len(self.mb.models) == 2)

        second_model = Model("second_model")
        self.mb.add(second_model)
        with self.subTest("override existing model"):
            self.assertTrue(len(self.mb.models) == 2)
        with self.subTest("added model under new name"):
            self.assertNotEqual(self.mb.models["second_model"], self.second_model)

        self.mb.add(second_model, name="first_model")
        with self.subTest("added same model under new name"):
            self.assertEqual(self.mb.models["first_model"], self.mb.models["second_model"])

    def test_drop(self):
        droped_model = self.mb.drop(name="first_model")
        with self.subTest("drop model"):
            self.assertTrue(len(self.mb.models) == 0)
        with self.subTest("droped model is model"):
            self.assertEqual(droped_model.name, self.first_model.name)
        with self.subTest("exception if model not exists"):
            with self.assertRaises(KeyError):
                self.mb.drop(name="first_model")

    def test_drop_all(self):
        self.mb.add(self.second_model)
        self.mb.drop_all()
        with self.subTest("drop model"):
            self.assertTrue(len(self.mb.models) == 0)

    def test_get(self):
        get_model = self.mb.get("first_model")
        with self.subTest("is expected model"):
            self.assertEqual(get_model.name, self.first_model.name)
        with self.subTest("exception if model not exists"):
            with self.assertRaises(KeyError):
                self.mb.get(name="no_model")

    def test_list_models(self):
        expected_set = {"first_model", "second_model"}
        self.mb.add(self.second_model)
        self.assertTrue(set(self.mb.list_models()) == expected_set)


    @unittest.skip("not build yet")
    def test_execute(self):
        pass

    @unittest.skip("not build yet")
    def test_upload_files(self):
        pass

    @unittest.skip("not build yet")
    def test_extractModelByStatement(self):
        pass

    @unittest.skip("not build yet")
    def test_extractFrom(self):
        pass

    @unittest.skip("not build yet")
    def test_extractDifferenceTo(self):
        pass

    @unittest.skip("not build yet")
    def test_extractShow(self):
        pass

    @unittest.skip("not build yet")
    def test_extractSplitBy(self):
        pass

    @unittest.skip("not build yet")
    def test_extractModel(self):
        pass

    @unittest.skip("not build yet")
    def test_extractPredict(self):
        pass

    @unittest.skip("not build yet")
    def test_extractDefaultValue(self):
        pass

    @unittest.skip("not build yet")
    def test_extractDefaultSubset(self):
        pass

    @unittest.skip("not build yet")
    def test_extractHide(self):
        pass

    @unittest.skip("not build yet")
    def test_extractAs(self):
        pass

    @unittest.skip("not build yet")
    def test_extractWhere(self):
        pass

    @unittest.skip("not build yet")
    def test_extractOpts(self):
        pass

    @unittest.skip("not build yet")
    def test_extractReload(self):
        pass

    @unittest.skip("not build yet")
    def test_extractSelect(self):
        pass


if __name__ == '__main__':
    unittest.main()
