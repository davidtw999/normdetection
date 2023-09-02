from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import NormDectector

class NormDetectorView(APIView):
  permissions_classes = [IsAuthenticated]
  def post(self, request, *args, **kwargs):
    print(request.data)
    request_data = {
        'asr_text': request.data.get('asr_text'),
        'uuid': request.data.get('uuid')
    }
    # TODO: add authorization
    model = NormDectector.load()
    output = model.detect_norm(request_data)
    return Response(output.to_json(), status=status.HTTP_200_OK)
