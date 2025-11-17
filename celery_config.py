# celery_config.py
from celery.schedules import crontab

beat_schedule = {
    'run-prediction-cycle-every-15-mins': {
        'task': 'worker.start_engine_loop',
        'schedule': crontab(minute='*/15'),  # This means run at :00, :15, :30, :45 of every hour
    },
}