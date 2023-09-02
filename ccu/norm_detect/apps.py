from django.apps import AppConfig


class NormDetectConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'norm_detect'

    def ready(self):
        print("Run ready")
        from .models import NormDectector
        NormDectector.load()