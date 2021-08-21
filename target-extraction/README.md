# Target extraction

In this package, it is possible to train your own extraction features model with the same structure as the [LayoutLM](https://huggingface.co/microsoft/layoutlm-base-uncased) model.
Disposing of invoice templates, you can generate your dataset with the following [repository](https://github.com/h2o64/faktur_generator). A pre-trained model is available in the 
**models** directory. You can also use directly the code **target_text_extraction.py** to extract with the pre-trained model the text from an invoice. The result is saved in json 
file in the specified directory. This model has been pre-trained on a polish invoices dataset, using the repository quoted, based on 20 templates with 1 000 generated invoices per 
template.

# Train your LayoutLM model

## Installation & Usage

First you need to clone 2 gits and install several tools :

~~~
! git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git
! cd unilm/layoutlm
! pip install unilm/layoutlm
~~~

and 

~~~
! git clone https://github.com/huggingface/transformers.git
! cd transformers
! pip install ./transformers
~~~

