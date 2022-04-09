import argparse
import importlib.util

from src.common.decorators import classproperty


class SettingsModuleLoader(object):

    @classmethod
    def parse_settings_arg(cls):
        def parse_args_for_settings_filename():
            # name of settings file to use must be passed as an argument to the program
            parser = argparse.ArgumentParser(description='Determine settings file')
            parser.add_argument('--settings', type=str, help='settings file to use')
            args = parser.parse_args()

            if not args.settings:
                raise ValueError(
                    "Name of settings file to be used must be passed as input. To run: python3 main.py --settings settings-whatever.py")

            return args.settings

        return parse_args_for_settings_filename()

    @classmethod
    def load_settings_from_file(cls, settings_filename):
        try:
            print("Loading settings from: ", settings_filename)
            spec = importlib.util.spec_from_file_location('cfg', settings_filename)
            settings_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings_module)
            return settings_module
        except:
            raise RuntimeError("FAILED! Unable to load settings file.")

    @classmethod
    def init_settings(cls):
        cls._settings_file = cls.parse_settings_arg()
        settings_module = cls.load_settings_from_file(cls._settings_file)
        settings_module.Settings.ensure_all_settings_initialized()

    @classproperty
    def settings_file(cls):
        return cls._settings_file
