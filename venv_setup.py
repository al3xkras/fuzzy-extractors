import os
import platform
import shutil

versions = ["3.10"]
prompt = False
venv_path = "./venv"
python_path = "./venv/Scripts/python"
pip_path = "./venv/Scripts/pip"


class Venv:
    def __init__(self, pip_abspath, python_abspath):
        self.pip_abspath = pip_abspath
        self.python_abspath = python_abspath

    def python(self, cmd: str) -> int:
        return os.system("%s %s" % (self.python_abspath, cmd))

    def pip(self, cmd: str) -> int:
        return os.system("%s %s" % (self.pip_abspath, cmd))


def prompt_Yn(msg: str) -> bool:
    """
    :param msg:
    :return: true if Y else false
    """
    if not prompt:
        return True
    s = input(msg + " (Y/n): ")
    if not s:
        return True
    return s[0].lower() == "y"


def prompt_yN(msg: str) -> bool:
    """
    :param msg:
    :return: true if Y else false
    """
    if not prompt:
        return False
    s = input(msg + " (y/N): ")
    if not s:
        return False
    return s[0].lower() == "y"


def python_interpreter_cmd() -> str:
    if any(x in os.popen("python --version").read() for x in versions):
        return "python"
    elif any(x in os.popen("python3 --version").read() for x in versions):
        return "python3"
    raise Exception("required python interpreter version not found: %s")


def check_venv_exists() -> bool:
    venv_p = os.path.abspath(venv_path)
    if os.path.exists(venv_p) and \
            prompt_yN("./venv path exists. Should remove?"):
        shutil.rmtree(venv_p)
        print("venv removed")
        return False
    return os.path.exists(venv_p)


def check_python_version():
    ver = platform.python_version()
    r = any(x in ver for x in versions)
    if not r:
        raise Exception("Python %s is not supported" % ver)


def install_required_packages():
    req_path = "./requirements.txt"
    cmd = "pip install -r %s" % os.path.abspath(req_path)
    if not os.system(cmd) == 0:
        raise Exception("requirements installation failed")


def create_venv():
    python = python_interpreter_cmd()
    cmd = "%s -m venv %s" % (python, os.path.abspath(venv_path))
    if not os.system(cmd) == 0:
        raise Exception("failed to create venv")
    print("venv created")


def init_venv(skip:bool):
    check_python_version()
    if skip and check_venv_exists():
        return Venv(python_abspath=os.path.abspath(python_path),
                    pip_abspath=os.path.abspath(pip_path))
    if not check_venv_exists():
        create_venv()
    else:
        print("venv exists")
    install_required_packages()
    return Venv(python_abspath=os.path.abspath(python_path),
                pip_abspath=os.path.abspath(pip_path))


def venv_setup_windows(skip:bool) -> Venv:
    return init_venv(skip)


def venv_setup_linux(skip:bool) -> Venv:
    return init_venv(skip)


def venv_setup(skip=False) -> Venv:
    system = platform.system().lower()
    if system.startswith("win"):
        return venv_setup_windows(skip)
    elif system.startswith("linux"):
        return venv_setup_linux(skip)
    else:
        raise Exception("OS is not supported: %s" % system)


if __name__ == '__main__':
    venv_setup()
