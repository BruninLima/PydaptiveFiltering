import os
import inspect
import importlib
import pkgutil
import pydaptivefiltering

def discover_adaptive_filters():
    print(f"{'Classe':<20} | {'Parâmetros do __init__':<60}")
    print("-" * 85)
    
    # Caminho base do pacote
    package_path = os.path.dirname(pydaptivefiltering.__file__)
    
    # Varre todos os módulos dentro do pacote
    for loader, module_name, is_pkg in pkgutil.walk_packages([package_path], pydaptivefiltering.__name__ + "."):
        try:
            module = importlib.import_module(module_name)
            
            # Procura por classes dentro do módulo
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Filtra apenas classes que herdam de AdaptiveFilter (ajuste o nome se necessário)
                # ou que foram definidas no próprio pacote (evita imports externos)
                if obj.__module__.startswith("pydaptivefiltering"):
                    
                    # Pega a assinatura do método __init__
                    try:
                        signature = inspect.signature(obj.__init__)
                        params = []
                        for param_name, param in signature.parameters.items():
                            if param_name != 'self':
                                # Formata como 'nome=default' se houver
                                default = f"={param.default}" if param.default is not inspect.Parameter.empty else ""
                                params.append(f"{param_name}{default}")
                        
                        params_str = ", ".join(params)
                        print(f"{name:<20} | {params_str[:60]}")
                    except Exception:
                        continue
        except Exception as e:
            # print(f"Erro ao carregar módulo {module_name}: {e}")
            continue

if __name__ == "__main__":
    discover_adaptive_filters()