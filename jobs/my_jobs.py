from apscheduler.schedulers.background import BackgroundScheduler
from .models import generate_model

scheduler = BackgroundScheduler()

seconds_in_day = 60 * 60 * 23

scheduler.add_job(generate_model, 'interval', seconds=seconds_in_day)