from venv_setup import venv_setup, Venv

if __name__ == '__main__':
    venv = venv_setup(True)
    venv.python("test_venv.py")
