#python -m llava.serve.controller --host 0.0.0.0 --port 10001 &
#mkdir -p logs
#python -m llava.serve.gradio_web_server --controller http://localhost:10001 --model-list-mode reload >> logs/gradio.log 2>&1 &
model_path=/comp_robot/zhanghao/model/llava_stage2_new_flickr_mark/
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10001 --port 40001 --worker http://localhost:40001 --model-path $model_path >> logs/model_worker.log 2>&1 &

#python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>