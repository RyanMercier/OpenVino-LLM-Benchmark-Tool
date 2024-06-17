OpenVino Installation and Benchmark Instructions

Notes:
openvino_benchmark.py - this is a script which queries prompts to an llm and displays performance metrics
ov_model - this is the directory in which the llm model should be stored in OpenVinoIR format (tested with llama3)



Usage:

vinoenv\Scripts\activate
pip install -r requirements.txt
python openvino_benchmark.py --device="GPU"



NOTE If you are using a custom model you must convert it to OpenVinoIR format:
	This works for pt, safetensors, tf, and onnx formats (tested with safetensors)
	You can clone and convert models hosted on huggingface in one step by using it's model_id instead of a local path

General case
	optimum-cli export openvino --model <model_id_or_path> --task <task> <out_dir>

Example
	optimum-cli export openvino --model <path_to_model_directory> --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 --sym ov_model

	More info
	https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html
	https://huggingface.co/docs/optimum/intel/inference


For more information: https://docs.openvino.ai/2024/get-started.html
					  https://huggingface.co/docs/optimum/index
					  https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html
					  https://huggingface.co/docs/optimum/main/en/intel/reference_ov

