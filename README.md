# deep_learning

# To start using our model:
install torch + torchvision : "pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html"


or:

"pip install torch"

"pip install torchvision"


# You need to verify that:
In the folder of model.py found the following folders: default_models, client_saved_model, Masks, predictions.

### client_saved_model - will contain the client models.

default models - contain our trained models.

Masks - contain masks for the model (for training).

prediction - will contain the predictions of the clients photo.




### To run enter "python3 model.py"

if you have gpu the model will use it for learnining.
