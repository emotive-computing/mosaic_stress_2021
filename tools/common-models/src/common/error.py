from src.common.singleton import SingletonFactory, Singleton


class SettingsConfigurationError(Exception):
    def __init__(self, setting, value, expected):
        self.message = \
            "Error. Settings file configuration for property {} was set to have a value of {}, " \
            "but must have type of {}.".format(setting, value, expected)
    def __str__(self):
        return self.message

class MissingSettingError(Exception):
    def __init__(self, setting, expected):
        self.message = "Error. Settings file configuration for property {} was not found, but this is a required property. It must have type of {}.".format(setting, expected)
    def __str__(self):
        return self.message

class AdditionalSettingError(Exception):
    def __init__(self, setting):
        self.message = "Error. Settings file configuration declared property {}, but this is currently not an expected property.".format(setting)
    def __str__(self):
        return self.message

class SettingsMismatchError(Exception):
    def __init__(self, setting1, setting2, value1, value2):
        self.message = "Error. Settings file configuration for properties {} and {} are set to have a value of {} and {}, but these are not allowed to both be set to these values.".format(setting1, setting2, value1, value2)
    def __str__(self):
        return self.message

class SettingsGenericConfigurationError(Exception):
    def __init__(self, setting, value, message_details):
        self.message = "Error. Settings file configuration for property {} with a value of {} is not a valid configuration. Details: {}".format(setting, value, message_details)
    def __str__(self):
        return self.message

class MissingLabelEncoderException(Exception):
    def __init__(self, label):
        self.message = "Tried to get existing label encoder for label: {}, but it does not exist. Make sure previous incremental predictions are handled for all required labels.".format(label)
    def __str__(self):
        return self.message



class ShowDeprecationWarning():

    labels = {}

    @classmethod
    def display(cls, label, message):
        if label not in cls.labels:
            print(DeprecationWarning(message))
            cls.labels[label] = True


