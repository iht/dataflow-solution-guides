#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
import time
from typing import Sequence, Optional, Any, Iterable

import vertexai
from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.metrics import Metrics
from apache_beam.ml.inference.base import ModelHandler, PredictionResult
from apache_beam.utils import retry
from google.api_core.exceptions import TooManyRequests, ServerError
from vertexai.preview.vision_models import ImageGenerationModel


def _retry_on_appropriate_gcp_error(exception):
  """
  Retry filter that returns True if a returned HTTP error code is 5xx or 429.
  This is used to retry remote requests that fail, most notably 429
  (TooManyRequests.)

  Args:
    exception: the returned exception encountered during the request/response
      loop.

  Returns:
    boolean indication whether or not the exception is a Server Error (5xx) or
      a TooManyRequests (429) error.
  """
  return isinstance(exception, (TooManyRequests, ServerError))


class VertexAIImagenModelHandler(ModelHandler[str, PredictionResult, ImageGenerationModel]):
  LOGGER = logging.getLogger("VertexAIImagenModelHandler")

  def __init__(self,
               model_name: str = "vertex_ai_imagen",
               imagen_model: str = "imagegeneration@006",
               **kwagrs_vertexai_params):
    """ Implementation of the ModelHandler interface for Imagen to generate images using text as input.

    Example Usage::

      pcoll | RunInference(VertexAIImagenModelHandler())

    Args:
      model_name: The Imagen model name. Default is vertex_ai_imagen.
      imagen_model: The Imagen model version. Default is imagegeneration@006.
    """
    super().__init__()
    self._model_name = model_name
    self._imagen_model = imagen_model
    self._env_vars = {}
    self.kwagrs_vertexai_params = kwagrs_vertexai_params

    self.throttled_secs = Metrics.counter(
      VertexAIImagenModelHandler, "cumulativeThrottlingSeconds")
    self.throttler = AdaptiveThrottler(window_ms=1, bucket_ms=1, overload_ratio=2)

  def load_model(self) -> ImageGenerationModel:
    """Loads and initializes a model for processing."""
    vertexai.init(**self.kwagrs_vertexai_params)
    model = ImageGenerationModel.from_pretrained(self._imagen_model)
    return model

  @retry.with_exponential_backoff(
    num_retries=5, retry_filter=_retry_on_appropriate_gcp_error)
  def _get_request(
      self,
      example: Any,
      model: ImageGenerationModel,
      throttle_delay_secs: int,
      inference_args: Optional[dict[str, Any]]):

    MSEC_TO_SEC = 1000
    while self.throttler.throttle_request(time.time() * MSEC_TO_SEC):
      self.LOGGER.info(
        "Delaying request for %d seconds due to previous failures",
        throttle_delay_secs)
      time.sleep(throttle_delay_secs)
      self.throttled_secs.inc(throttle_delay_secs)

    try:
      req_time = time.time()
      prediction = model.generate_images(**example)
      self.throttler.successful_request(req_time * MSEC_TO_SEC)
      return prediction
    except TooManyRequests as e:
      self.LOGGER.warning("request was limited by the service with code %i", e.code)
      raise
    except Exception as e:
      self.LOGGER.error("unexpected exception raised as part of request, got %s", e)
      raise

  def run_inference(
      self,
      batch: Sequence[str],
      model: ImageGenerationModel,
      inference_args: Optional[dict[str, Any]] = None
  ) -> Iterable[PredictionResult]:
    """Runs inferences on a batch of text strings.

    Args:
      batch: A sequence of examples as text strings.
      model: The ImageGenerationModel being used.
      inference_args: Any additional arguments for an inference.

    Returns:
      An Iterable of type PredictionResult.
    """
    predictions = []
    for one_text in batch:
      images = self._get_request(one_text, model, throttle_delay_secs=5,
                                 inference_args=inference_args)
      for img in images:
        yield PredictionResult(one_text, img, self._model_name)
