# strikegen
Generalization measures of Language Models

## Usage
---
### Sharpness

The following code can be used to get an idea of how to calculate model sharpness.

```py
from strikegen.sharpness import Sharpness

# initialize sharpness object
strike_sharpness = Sharpness()
# get batch from data
batch_0 = eval_dataloader[0]
# offload batch on gpu (if any)
batch_0 = tuple(t.to(device) for t in batch)
# convert the batch to dict
inputs = {
    "input_ids": batch_0[0],
    "attention_mask": batch_0[1],
    "labels": batch_0[3]
}

# For more robust calculation, we can average the sharpness over k steps
# but for showing the usecase, lets just calculate directly

# get the model and tokenizer
model, tokenizer = get_model()

# calculate sharpness of a model on a given batch
sharpness = strike_sharpness.calculate(
    model=model,
    input_tensor=inputs,
    label_tensor=inputs["labels"],
    gradient_accumulation_steps=gradient_accumulation_steps,
    noise_scale=noise_scale
)

```
---
### Margin

```py
from strikegen.margin import Margin
strike_margin = Margin()

# get batch from data
batch_0 = eval_dataloader[0]
# offload batch on gpu (if any)
batch_0 = tuple(t.to(device) for t in batch)
# convert the batch to dict
inputs = {
    "input_ids": batch_0[0],
    "attention_mask": batch_0[1],
    "labels": batch_0[3]
}

# get the model and tokenizer
model, tokenizer = get_model()
# get model outputs
x = model(**inputs)
# get softmax output
with torch.no_grad():
    probabilities = nn.functional.softmax(x.logits, dim=-1)

# margin for batch_0
margin = strike_margin.calculate(probabilities=probabilities,
                        true_label_indexes=inputs['labels'],
                        margin_type='mean'
)

```