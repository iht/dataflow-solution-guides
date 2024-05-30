./gradlew run -Pargs="
  --pipeline=SPANNER_CHANGE_STREAM \
  --streaming \
  --enableStreamingEngine \
  --autoscalingAlgorithm=THROUGHPUT_BASED \
  --runner=DataflowRunner \
  --experiments=use_runner_v2 \
  --project=$PROJECT \
  --tempLocation=$TEMP_LOCATION \
  --region=$REGION \
  --serviceAccount=$SERVICE_ACCOUNT \
  --subnetwork=$NETWORK \
  --experiments=enable_data_sampling;use_network_tags=ssh;dataflow \
  --usePublicIps=false \
  --pubsubTopic=$TOPIC \
  --spannerInstance=$SPANNER_INSTANCE \
  --spannerDatabase=$SPANNER_DATABASE \
  --spannerChangeStream=$SPANNER_CHANGE_STREAM \
  --pubsubOutputTopicCount=$PUBSUB_TOPIC_COUNT"