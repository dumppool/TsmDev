coverage run  tf_cnn_benchmarks.py --num_gpus=2 --batch_size=32 --model=resnet50 --variable_update=parameter_server
coverage html -d covhtml
rm covhtml/_*


