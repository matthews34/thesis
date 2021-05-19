import logging

class Formatter(logging.Formatter):
    error_fmt = "%(asctime)s ERROR: %(msg)s"
    dubug_fmt = "%(asctime)s DEBUG: %(module)s:%(lineno)d: %(msg)s"
    info_fmt = "%(asctime)s %(msg)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S',)
    
    def format(self, record):

        original_fmt = self._style._fmt

        if record.levelno == logging.DEBUG:
            self._style._fmt = Formatter.dubug_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = Formatter.info_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = Formatter.error_fmt

        result = logging.Formatter.format(self, record)

        self._fmt = original_fmt

        return result