# NLP

# 1. Bert as a service
This is actually is something I did half a year ago.
The original bert is a file based processor which read all input at the same time. However there is need that the input coming online, the data only available when the request arrives. So I revise bert program to make it could be a standby service to process online requests.

bert_client() is the code for your own logic.
