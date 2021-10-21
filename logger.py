class Logger:
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with open(f'{self.name}.txt', 'a') as f:
                f.write(f'{func(*args, **kwargs)}')
                f.write('\n')
        return wrapper
