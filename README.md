Summary:

In this project I implemented the model presented in the Deep Learning
of Binary Hash Codes for Fast Image Retrieval paper.

In addition to the paper, I've Implemented the model using several
transfer learning nets and then I examined the differences between the
scores and linkage between the test set accuracy to the image retrieval
scores.

Running Instructions:

1. Net training:
    To train the net, you need to run the following command:
    python train.py --name [model name] --lr [learning rate] --
    momentum [momentum] --epoch [number of epochs] --batch
    [batch size] --bits [number of bits] --path [folder path] --mid
    [transfer learning net]
2. Run Image Retrieval:
    To run the retrieval, you need to run the following command:
    python run_retrieval.py --batch [batch size] --bits [number of bits]
    --path [folder path] --mid [transfer learning net] --rand [number of
    random images per class]

**Default values:**

[model name] = 'my_model.pkl'

[learning rate] = 0.

[momentum] = 0.

[number of epochs] = 50

[batch size] = 200

[number of bits] = 48

[folder path] = ''

[transfer learning net] = 'AlexNet'

[number of random images per class] = 0, means all images

**Remarks:**

[transfer learning net] needs to be one of the following:

{'AlexNet', 'VGG19','VGG16','ResNet18','GoogLeNet'}

Full report is on report.pdf file