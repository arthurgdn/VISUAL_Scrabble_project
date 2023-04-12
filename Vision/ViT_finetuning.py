from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
import cv2
from datasets import load_metric
from Vision.dataset_creation import create_dataset
from Vision.dataset_creation import transform
from Vision.dataset_creation import compute_metrics

extractor = AutoFeatureExtractor.from_pretrained("pittawat/vit-base-letter")

model = AutoModelForImageClassification.from_pretrained("pittawat/vit-base-letter")

metric = load_metric("accuracy")
img_1 = cv2.imread(r'data\I.jpg',0)
img_2 = cv2.imread(r'data\I2.jpg',0)
list_img = [img_1, img_2, img_1]*20
ds2 = create_dataset(list_img)
prepared_ds2 = list(ds2.with_transform(lambda x : transform(x, extractor)))

training_args = TrainingArguments(
 output_dir="./vit-base-beans",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=2,
  learning_rate=5e-5,
)


trainer = Trainer(
    model=model,
    args=training_args,
    #data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds2,
    #eval_dataset=prepared_ds["validation"],
    #tokenizer=extractor,
)

train_results = trainer.train()
# trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()
torch.save(model, 'model/vit-ft.pt')