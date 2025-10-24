import openfhe
from openfhe import *
import inspect

members = inspect.getmembers(openfhe)

my_classes_dict = dict()

with open('typings/openfhe.pyi', 'w+') as f: 
    
    for name, obj in members:
        if name.startswith("_"):
            continue

        if inspect.isclass(obj):
            f.write(f"class {name}:\n")
            cls = getattr(openfhe, name)
            for m in dir(cls):
                if not m.startswith('_'):
                    f.write(f"\t{m}: Any\n")
            f.write("\n\t...\n")
                    
         
            if name not in my_classes_dict:
                my_classes_dict[name] = list()

        elif inspect.isfunction(obj):
            f.write(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...\n")
        else:
            if type(obj).__name__ in my_classes_dict:
                my_classes_dict[type(obj).__name__].append(name)
            elif type(obj).__name__ == 'builtin_function_or_method':
                f.write(f"{name}: {type(obj).__name__}\n")
            else: 
                my_classes_dict[type(obj).__name__] = [name]