# Autoencoder-FraudDetection

To detect anomaly, or outlier, especially under the unsupervised environment, we need a simple-to-use model robust to any set of features and free of assumptions.

We pick Autoencoder chiefly because the structure of Autoencoder itself is well fit for performing this particular task. Anomaly detection (or Fraud detection at large) is often centred at whether the new data violates the model of identity function 𝑓(𝑥) = 𝑥. The idea behind Autoencoder is essentially to build such a f to replicate itself. To extract the relationship among existing data, we then know how the new data would be considered as outliers based on what the model had learnt from the old data.

Autoencoder is composed of two main parts: Encoder and Decoder. The basic idea is this: Encoder tries to learn the incoming data, then he has to represent what he thinks it is through the middle layer called Bottleneck, through which Encoder communicates what he has learnt to Decoder. Decoder, who has no direct access to data, has to reconstruct what it should happen as the full picture of the data. Model performance relies on how accurate this reconstruction was, and we evaluate by the measure called (Squared) Reconstruction Error. 𝐿(𝑥, 𝑥′) = ||𝑥 − 𝑥′||^2
