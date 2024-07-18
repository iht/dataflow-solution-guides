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
import base64
import hashlib
import os.path
import typing

import apache_beam as beam
from apache_beam import Pipeline, PCollection
from apache_beam.io.gcp import pubsub
from apache_beam.ml.inference import RunInference
from apache_beam.ml.inference.base import PredictionResult

from .model_handlers import VertexAIImagenModelHandler
from .options import MyPipelineOptions


class PromptAndPaths(typing.NamedTuple):
  prompt: str
  paths: list[str]

def _format_output(element: PredictionResult) -> str:
  return "Input: \n{input}, \n\n\nOutput: \n{output}".format(
    input=element.example, output=element.inference)


@beam.ptransform_fn
def _extract(p: Pipeline, subscription: str) -> PCollection[str]:
  msgs: PCollection[bytes] = p | "Read subscription" >> beam.io.ReadFromPubSub(
    subscription=subscription)
  return msgs | "Parse" >> beam.Map(lambda x: x.decode("utf-8"))


@beam.ptransform_fn
def _transform(msgs: PCollection[str]) -> PCollection[str]:
  preds: PCollection[PredictionResult] = msgs | "RunInference-Gemma" >> RunInference(
    VertexAIImagenModelHandler())
  return preds | "Format Output" >> beam.Map(_format_output)

@beam.ptransform_fn
def _load(imgs: PCollection[PredictionResult], save_location: str) -> PCollection[str]:
  def _save_and_get_details(p: PredictionResult) -> PromptAndPaths:
    prompt = p.example
    # Hash of prompt
    hasher = hashlib.sha1(usedforsecurity=False)
    hasher.update(prompt.encode('utf-8'))
    hash_value = hasher.hexdigest()
    fns = []
    for idx, img in enumerate(p.inference):
      filename = os.path.join(save_location, f"{hash_value}_{idx}.png")
      img.save(filename, include_generation_parameters=False)
      fns.append(filename)
    return PromptAndPaths(prompt=prompt, paths=fns)

  reshuffled: PCollection[PredictionResult] = imgs | "Reshuffle" >> beam.Reshuffle



def create_pipeline(options: MyPipelineOptions) -> Pipeline:
  """ Create the pipeline object.

  Args:
    options: The pipeline options, with type `MyPipelineOptions`.

  Returns:
    The pipeline object.
    """
  pipeline = beam.Pipeline(options=options)
  # Extract
  msgs: PCollection[str] = pipeline | "Read" >> _extract(subscription=options.messages_subscription)
  # Transform
  responses: PCollection[str] = msgs | "Transform" >> _transform()
  # Load
  responses | "Publish Result" >> pubsub.WriteStringsToPubSub(topic=options.responses_topic)

  return pipeline
