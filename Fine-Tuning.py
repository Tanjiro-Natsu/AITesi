import csv
def formated_prompt(prompt,liked):
 return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n You're a movies recommendder.....<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{liked}<|ieot_id|>"


