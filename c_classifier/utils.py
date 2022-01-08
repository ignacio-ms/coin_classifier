import time


def timed(fun):
    def wrapper(*args, **kwargs):
        before = time.time()
        res = fun(*args, **kwargs)
        after = time.time()

        f_name = fun.__name__
        print(f'{f_name} took {after - before}[s] to execute.\n')
        return res

    return wrapper
