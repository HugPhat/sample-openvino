import functools


def inspect(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        print(kwargs.get('arg'))
        abc = kwargs.get('arg')
        value = func(abc)

        print('end')
        print('of')
        print('the code')

        return value
    return wrapper


@inspect
def my_func(arg):
    # Do something
    print("my_func is called with arg =", arg)


my_func(arg = 'abcn')
