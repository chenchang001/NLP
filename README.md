# NLP

# 1. Bert as a service
This is actually is something I did half a year ago.
The original bert is a file based processor which read all input at the same time. However there is need that the input coming online, the data only available when the request arrives. So I revise bert program to make it could be a standby service to process online requests.

bert_client() is the code for your own logic.

# 2. Compare XLNet with Bert
Use XLNet on the sentimental analysis with same train/dev dateset.
Comparson result is attached. 0.88 vs 0.87
Background: XLNet outperforms BERT on several NLP Tasks since its release...
