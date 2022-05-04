# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
import glob
import os
import importlib


class MetaModule(type):

    def __get_module_dict(cls):
        # Getting the current path, the file path and the module directory path
        cwd_path = os.getcwd()
        file_path = os.path.dirname(__file__)

        os.chdir(file_path)
        import_module_list = glob.glob("*.py")
        import_module_list.remove("models.py")
        for import_module in import_module_list:
            import_module = import_module.replace(".py", "")
            import_module = "."+import_module
            importlib.import_module(import_module, package="models")

        # Setting back the old current directory
        os.chdir(cwd_path)

        module_dict = {}
        for module in sys.modules:
            if(re.match(r"^models[.].+", module)):
                module_class = sys.modules[module].Module
                module = module.replace("models.", "")
                module_dict[module] = module_class
        return module_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the module
        module_name = args[0]

        # Getting the module dictionnary
        module_dict = cls.__get_module_dict()

        # Checking that the module exists
        if(module_name not in module_dict):
            raise Exception(module_name+" doesn't exist")

        # Adding the new module in the base classes
        bases += (module_dict[module_name], )

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaModule, new_cls).__call__(*args, **kwargs)


class Module(metaclass=MetaModule):
    '''
    This is the class ``Module''.

    This is the class that you call to instantiate any models
    by specifying the right name.

    Parameters
    ----------
    name: string
        The name of the model. Could be one among the following list
        -> (forest_model)
    device: device
        The device you work on (e.g. cuda or cpu).
    '''
    def __init__(self, name, device, **kwargs):
        super().__init__(device, **kwargs)
