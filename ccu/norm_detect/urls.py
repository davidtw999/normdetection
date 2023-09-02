
from django.urls import path
from rest_framework.authtoken import views

from .views import NormDetectorView

urlpatterns = [
  path('norm-detect', NormDetectorView.as_view(), name='norm-detect'),
  path('api-token-auth/', views.obtain_auth_token)
]
