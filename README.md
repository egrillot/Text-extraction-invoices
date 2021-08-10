Text-extraction-invoices
=====================

The objective is to be able to extract the text from an invoice while maintaining its structure. 
The code also performs well for a poor quality invoice by using an upscaling model developed [here](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/202-vision-superresolution).

Installation
----

The easiest way to install is using the following command line : must 
~~~ 
!python3 -m pip install --ignore-installed -r requirements.txt 
~~~

Usage
----

Three inputs must be given :

``file_path`` : the path of the invoice. The format must be readable by the Pillow library or a .pdf

``dir_result`` : the path of the directory to save the result wich is a .txt with the texts and their coordinates

``models_path`` : the path of the models folder

Then you can use the following command to extract the text from your invoice :

~~~ 
!python3 text_extraction.py --file_path=your/invoice/path --dir_result==directory/to/save/the/result --models_path==path/to/the/models/folder
~~~

