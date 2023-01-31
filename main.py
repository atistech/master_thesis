import datasets
import utils
import threading

dataset = datasets.MnistDataset()
model_texts = utils.sample_model_texts()

threads = []
for model_text in model_texts:
  thread = threading.Thread(target=utils.model_run, args=(dataset, model_text, utils.text_to_model(model_text),))
  threads.append(thread)

for thread in threads:
  thread.start()

for thread in threads:
  thread.join()

#for a in agents:
#  print(a.accuracy_results)