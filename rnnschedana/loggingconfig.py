import logging
import logging.config



def getLogger(name):
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(levelname)-8s %(asctime)s - %(name)-20s: %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'},
            'notime': {'format': '%(levelname)-8s %(name)-20s: %(message)s'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'notime': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'notime',
                'stream': 'ext://sys.stdout'
            }
            #'file': {
            #   'level': 'DEBUG',
            #   'class': 'logging.handlers.RotatingFileHandler',
            #   'formatter': 'default',
            #   'filename': log_path,
            #   'maxBytes': 1024,
            #   'backupCount': 3
            #}
        },
        'loggers': {
            '': {
                'level': 'INFO',
                'handlers': ['console']
            },
            'SCH': {
                'level': 'INFO',
                'handlers': ['notime'],
                'propagate': False
            },
            'DATAGEN': {
                'level': 'INFO',
                'handlers': ['notime'],
                'propagate': False
            },
            'INTERFACE': {
                'level': 'INFO',
                'handlers': ['notime'],
                'propagate': False
            }
        },
        'disable_existing_loggers': False
    })
    return logging.getLogger(name)
